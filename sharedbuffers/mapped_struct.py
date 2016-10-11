# -*- coding: utf-8 -*-
# cython: infer_types=True, profile=False, linetrace=False
# distutils: define_macros=CYTHON_TRACE=0
import struct
import array
import mmap
import numpy
import tempfile
import functools
import lz4
import cPickle
import os
import sys
import xxhash
import itertools

from chorde.clients.inproc import Cache

import cython

npuint64 = cython.declare(object, numpy.uint64)
npuint32 = cython.declare(object, numpy.uint32)

if cython.compiled:
    # Compatibility fix for cython >= 0.23, which no longer supports "buffer" as a built-in type
    buffer = cython.declare(object)  # lint:ok
    from types import BufferType as buffer

class ubyte(int):
    pass
uint8 = ubyte

class byte(int):
    pass
int8 = byte

class ushort(int):
    pass
uint16 = ushort

class short(int):
    pass
int16 = short

class int32(int):
    pass

class uint32(int):
    pass

class int64(int):
    pass

class uint64(int):
    pass

class float32(float):
    pass

class float64(float):
    pass
double = float64

@cython.cfunc
@cython.inline
def _likebuffer(buf):
    if type(buf) is buffer or type(buf) is bytearray or type(buf) is bytes or isinstance(buf, bytes):
        return buf
    else:
        return buffer(buf)

class mapped_frozenset(frozenset):
    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        all_int = 1
        for x in obj:
            if type(x) is not int:
                all_int = 0
                break
        if all_int:
            maxval = max(obj) if obj else 0
            minval = min(obj) if obj else 0
            if 0 <= minval and maxval < 120:
                # inline bitmap
                buf[offs] = 'm'
                buf[offs+1:offs+8] = '\x00\x00\x00\x00\x00\x00\x00\x00'
                for x in obj:
                    buf[offs+1+x/8] |= 1 << (x & 7)
                offs += 8
                return offs
            else:
                # Else, same representation as a tuple of sorted items, only backed in-memory by a frozenset
                tup = mapped_tuple(sorted(obj))
                return tup.pack_into(tup, buf, offs, idmap, implicit_offs)
        else:
            # Same representation as a tuple of items, only backed in-memory by a frozenset
            tup = mapped_tuple(obj)
            return tup.pack_into(tup, buf, offs, idmap, implicit_offs)

    @classmethod
    @cython.locals(i=int, j=int, offs=int, pybuf='Py_buffer', pbuf='const unsigned char *', b=cython.uchar)
    def unpack_from(cls, buf, offs, idmap = None):
        if cython.compiled:
            buf = _likebuffer(buf)
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)  # lint:ok
            pbuf = cython.cast(cython.p_uchar, pybuf.buf)  # lint:ok
        else:
            pbuf = buf
        try:
            if pbuf[offs] == 'm':
                # inline bitmap
                rv = []
                for i in xrange(7):
                    b = pbuf[offs+1+i]
                    if b:
                        for j in xrange(8):
                            if b & (1<<j):
                                rv.append(i*8+j)
                return cls(rv)
            else:
                # unpack a list, build a set from it
                return cls(mapped_list.unpack_from(buf, offs, idmap))
        finally:
            if cython.compiled:
                if type(buf) is buffer:
                    PyBuffer_Release(cython.address(pybuf))  # lint:ok

class mapped_tuple(tuple):
    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0,
            int = int, type = type, min = min, max = max, array = array.array, id = id):
        all_int = 1
        all_float = 1
        baseoffs = offs
        for x in obj:
            if type(x) is not int:
                all_int = 0
                break
        for x in obj:
            if type(x) is not float:
                all_float = 0
                break
        objlen = len(obj)
        if all_int:
            maxval = max(obj) if obj else 0
            minval = min(obj) if obj else 0
            if 0 <= minval and maxval <= 0xFF:
                # inline unsigned bytes
                buf[offs] = dtype = 'B'
            elif -0x80 <= maxval <= 0x7F:
                # inline signed bytes
                buf[offs] = dtype = 'b'
            elif 0 <= minval and maxval <= 0xFFFF:
                # inline unsigned shorts
                buf[offs] = dtype = 'H'
            elif -0x8000 <= maxval <= 0x7FFF:
                # inline signed shorts
                buf[offs] = dtype = 'h'
            elif 0 <= minval and maxval <= 0xFFFFFFFF:
                # inline unsigned shorts
                buf[offs] = dtype = 'I'
            elif -0x80000000 <= maxval <= 0x7FFFFFFF:
                # inline signed shorts
                buf[offs] = dtype = 'i'
            else:
                # inline sorted int64 list
                buf[offs] = 'q'
                dtype = 'l'
            if dtype == 'l':
                buf[offs+1:offs+8] = struct.pack('<Q', objlen)[:7]
                offs += 8
            elif objlen < 0xFFFFFF:
                buf[offs+1:offs+4] = struct.pack('<I', objlen)[:3]
                offs += 4
            else:
                buf[offs+1:offs+8] = '\xff\xff\xff\xff\xff\xff\xff'
                buf[offs+8:offs+12] = struct.pack('<Q', objlen)
                offs += 12
            a = array(dtype, obj)
            abuf = buffer(a)
            buf[offs:offs+len(abuf)] = abuf
            offs += len(abuf)
            offs = (offs + 7) / 8 * 8
            return offs
        elif all_float:
            buf[offs] = 'd'
            a = array('d', obj)
            buf[offs+1:offs+8] = struct.pack('<Q', objlen)[:7]
            offs += 8
            abuf = buffer(a)
            buf[offs:offs+len(abuf)] = abuf
            offs += len(abuf)
            offs = (offs + 7) / 8 * 8
            return offs
        else:
            # inline object tuple
            buf[offs] = 't'
            buf[offs+1:offs+8] = struct.pack('<Q', objlen)[:7]
            offs += 8
            
            # None will be represented with an offset of 1, which is an impossible offset
            # (it would point into this tuple's header, 0 would be the tuple itself so it's valid)
            indexoffs = offs
            index = array('l', [1]*len(obj))
            offs += len(buffer(index))

            if idmap is None:
                idmap = {}
            
            for i,x in enumerate(obj):
                if x is not None:
                    # these are wrapped objects, not plain objects, so make sure they have distinct xid
                    xid = id(x) | (0xFL << 64)
                    if xid not in idmap:
                        idmap[xid] = offs + implicit_offs
                        mx = mapped_object(x)
                        offs = mx.pack_into(mx, buf, offs, idmap, implicit_offs)
                    val_offs = idmap[xid]
                    index[i] = val_offs - baseoffs - implicit_offs

            # write index
            buf[indexoffs:indexoffs+len(buffer(index))] = buffer(index)
            
            return offs

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is None:
            idmap = {}
        if offs in idmap:
            return idmap[offs]
        rv = idmap[offs] = tuple(mapped_list.unpack_from(buf, offs, idmap))
        return rv

class mapped_list(list):
    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        # Same format as tuple, only different base type
        return mapped_tuple.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None, array = array.array, itemsizes = {
                dtype : array.array(dtype, []).itemsize
                for dtype in ('B','b','H','h','I','i','l','d')
            } ):
        if idmap is None:
            idmap = {}
        if offs in idmap:
            return idmap[offs]
        
        baseoffs = offs
        dcode = buf[offs]
        if dcode in ('B','b','H','h','I','i'):
            dtype = dcode
            objlen, = struct.unpack('<I', buf[offs+1:offs+4] + '\x00')
            offs += 4
            if objlen == 0xFFFFFF:
                objlen = struct.unpack_from('<Q', buf, offs)
                offs += 8
            rv = cls(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 'q':
            dtype = 'l'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8
            rv = cls(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 'd':
            dtype = 'd'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8
            rv = cls(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 't':
            dtype = 'l'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8

            index = array(dtype, buf[offs:offs+itemsizes[dtype]*objlen])

            rv = idmap[baseoffs] = cls([None] * objlen)
            for i,ix in enumerate(index):
                if ix != 1:
                    absix = ix + baseoffs
                    if absix in idmap:
                        rv[i] = idmap[absix]
                    else:
                        rv[i] = idmap[absix] = mapped_object.unpack_from(buf, absix, idmap)
        else:
            raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
        return rv

lz4_decompress = cython.declare(object, lz4.decompress)
lz4_compress = cython.declare(object, lz4.compress)

MIN_COMPRESS_THRESHOLD = cython.declare(cython.size_t, 512)

if cython.compiled:
    @cython.inline
    @cython.cfunc
    @cython.returns(bytes)
    @cython.locals(
        offs = cython.longlong, buflen = 'Py_ssize_t', objlen = cython.size_t, rv = bytes,
        pbuf = 'const char *', obuf = 'const char *', compressed = cython.bint)
    def _unpack_bytes_from_cbuffer(pbuf, offs, buflen, idmap):
        if idmap is not None and offs in idmap:
            return idmap[offs]
    
        assert offs + cython.sizeof(cython.ushort) <= buflen
        obuf = pbuf
        pbuf += offs
        header = cython.cast('const _varstr_header *', pbuf)
        compressed = (header.shortlen & 0x8000) != 0
        if (header.shortlen & 0x7FFF) == 0x7FFF:
            assert offs + cython.sizeof('_varstr_header') <= buflen
            objlen = header.biglen
            pbuf += cython.sizeof('_varstr_header')
        else:
            objlen = header.shortlen & 0x7FFF
            pbuf += cython.sizeof(cython.ushort)
        # unconditional assert, for basic safety even when assertions are off
        # up to this point, garbage input can only cause segfaults, but here
        # they could cause huge memory allocations and DoS-type of issues
        if objlen > buflen:
            raise AssertionError
        assert (pbuf - obuf) + objlen <= buflen
        rv = PyBytes_FromStringAndSize(pbuf, objlen)  # lint:ok
        if compressed:
            rv = lz4_decompress(rv)
        
        if idmap is not None:
            idmap[offs] = rv
        return rv

    @cython.inline
    @cython.cfunc
    @cython.returns(cython.bint)
    @cython.locals(
        offs = cython.longlong, buflen = 'Py_ssize_t', objlen = cython.size_t, reflen = cython.size_t, rv = bytes,
        pbuf = 'const char *', obuf = 'const char *', refbuf = 'const char *', compressed = cython.bint)
    def _compare_bytes_from_cbuffer(refbuf, reflen, pbuf, offs, buflen):
        assert offs + cython.sizeof(cython.ushort) <= buflen
        obuf = pbuf
        pbuf += offs
        header = cython.cast('const _varstr_header *', pbuf)
        compressed = (header.shortlen & 0x8000) != 0
        if (header.shortlen & 0x7FFF) == 0x7FFF:
            assert offs + cython.sizeof('_varstr_header') <= buflen
            objlen = header.biglen
            pbuf += cython.sizeof('_varstr_header')
        else:
            objlen = header.shortlen & 0x7FFF
            pbuf += cython.sizeof(cython.ushort)
        if not compressed and objlen != reflen:
            return False
        # unconditional assert, for basic safety even when assertions are off
        # up to this point, garbage input can only cause segfaults, but here
        # they could cause huge memory allocations and DoS-type of issues
        if objlen > buflen:
            raise AssertionError
        assert (pbuf - obuf) + objlen <= buflen

        if not compressed:
            return memcmp(pbuf, refbuf, reflen) == 0 # lint:ok
        else:
            rv = PyBytes_FromStringAndSize(pbuf, objlen)  # lint:ok
            rv = lz4_decompress(rv)

            return reflen == len(rv) and memcmp(refbuf, cython.cast('const char *', rv), reflen) == 0 # lint:ok

@cython.cfunc
@cython.locals(
    offs = cython.longlong, objlen = cython.size_t,
    pbuf = 'const char *', pybuf='Py_buffer', compressed = cython.bint)
def _unpack_bytes_from_pybuffer(buf, offs, idmap):
    if idmap is not None and offs in idmap:
        return idmap[offs]

    if cython.compiled:
        try:
            buf = _likebuffer(buf)
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)  # lint:ok
            rv = _unpack_bytes_from_cbuffer(cython.cast(cython.p_char, pybuf.buf), offs, pybuf.len, None)  # lint:ok
        finally:
            PyBuffer_Release(cython.address(pybuf))  # lint:ok
    else:
        hpacker = struct.Struct('=H')
        objlen = hpacker.unpack_from(buf, offs)[0]
        dataoffs = offs + hpacker.size
        compressed = (objlen & 0x8000) != 0
        if (objlen & 0x7FFF) == 0x7FFF:
            qpacker = struct.Struct('=HQ')
            objlen = qpacker.unpack(buf, offs)[1]
            dataoffs = offs + qpacker.size
        rv = buffer(buf, dataoffs, objlen)
        if compressed:
            rv = lz4_decompress(rv)
        else:
            rv = bytes(rv)
    
    if idmap is not None:
        idmap[offs] = rv
    return rv

class mapped_bytes(bytes):
    @classmethod
    @cython.locals(
        offs = cython.longlong, implicit_offs = cython.longlong, 
        objlen = cython.size_t, objcomplen = cython.size_t, obj = bytes, objcomp = bytes,
        pbuf = 'char *', pybuf='Py_buffer', compressed = cython.ushort)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = id(obj)
            idmap[objid] = offs + implicit_offs
        objlen = len(obj)

        if objlen > MIN_COMPRESS_THRESHOLD:
            objcomp = lz4_compress(obj)
            objcomplen = len(objcomp)
            if objcomplen < (objlen - objlen/3):
                # Must get substantial compression to pay the price
                obj = objcomp
                objlen = objcomplen
                compressed = 0x8000
            else:
                compressed = 0
            del objcomp
        else:
            compressed = 0

        if (offs + 16 + len(obj)) > len(buf):
            raise struct.error('buffer too small')
        if cython.compiled:
            try:
                buf = _likebuffer(buf)
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_WRITABLE)  # lint:ok
                pbuf = cython.cast(cython.p_char, pybuf.buf) + offs  # lint:ok
                
                if objlen < 0x7FFF:
                    cython.cast('_varstr_header *', pbuf).shortlen = objlen | compressed
                    offs += cython.sizeof(cython.ushort)
                    pbuf += cython.sizeof(cython.ushort)
                else:
                    cython.cast('_varstr_header *', pbuf).shortlen = 0x7FFF | compressed
                    cython.cast('_varstr_header *', pbuf).biglen = objlen
                    offs += cython.sizeof('_varstr_header')
                    pbuf += cython.sizeof('_varstr_header')
                memcpy(pbuf, cython.cast(cython.p_char, obj), objlen)  # lint:ok
            finally:
                PyBuffer_Release(cython.address(pybuf))  # lint:ok
        else:
            if objlen < 0x7FFF:
                hpacker = struct.Struct('=H')
                hpacker.pack_into(buf, offs, objlen | compressed)
                offs += hpacker.size
            else:
                qpacker = struct.Struct('=HQ')
                qpacker.pack_into(buf, offs, 0x7FFF | compressed, objlen)
                offs += qpacker.size
            buf[offs:offs+objlen] = obj
        offs += objlen
        return offs

    @classmethod
    @cython.locals(
        offs = cython.longlong, objlen = cython.size_t,
        pbuf = 'const char *', pybuf='Py_buffer')
    def unpack_from(cls, buf, offs, idmap = None):
        return _unpack_bytes_from_pybuffer(buf, offs, idmap)
_mapped_bytes = cython.declare(object, mapped_bytes)

class mapped_unicode(unicode):
    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = id(obj)
            idmap[objid] = offs + implicit_offs

        return mapped_bytes.pack_into(obj.encode("utf8"), buf, offs, None, implicit_offs)

    @classmethod
    def unpack_from(cls, obj, buf, offs, idmap = None):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        rv = mapped_bytes.unpack_from(obj, buf, offs).decode("utf8")
        if idmap is not None:
            idmap[offs] = rv
        return rv

class mapped_object(object):
    __slots__ = ('value', 'typecode')

    TYPE_CODES = {
        uint8 : 'B',
        int8 : 'b',
        uint16 : 'H',
        int16 : 'h',
        uint32 : 'I',
        int32 : 'i',
        uint64 : 'Q',
        int64 : 'q',
        float64 : 'd',
        float32 : 'f',
        bool : 'T',

        mapped_frozenset : 'Z',
        mapped_tuple : 't',
        mapped_list : 'e',
        mapped_bytes : 's',
        mapped_unicode : 'u',
        mapped_bytes : 's',
        
        int : 'q',
        float : 'd',
        str : 's',
        unicode : 'u',
    }

    def p(s):
        return s, (s.size + 7) / 8 * 8 - s.size

    CODE_PACKER = p(struct.Struct('=c'))
    PACKERS = {
        'B' : p(struct.Struct('=cB')),
        'b' : p(struct.Struct('=cb')),
        'H' : p(struct.Struct('=cH')),
        'h' : p(struct.Struct('=ch')),
        'I' : p(struct.Struct('=cI')),
        'i' : p(struct.Struct('=ci')),
        'Q' : p(struct.Struct('=cQ')),
        'q' : p(struct.Struct('=cq')),
        'd' : p(struct.Struct('=cd')),
        'f' : p(struct.Struct('=cf')),
        'T' : p(struct.Struct('=c?')),
    }
    OBJ_PACKERS = {
        'Z' : (mapped_frozenset.pack_into, mapped_frozenset.unpack_from, mapped_frozenset),
        't' : (mapped_tuple.pack_into, mapped_tuple.unpack_from, mapped_tuple),
        'e' : (mapped_list.pack_into, mapped_list.unpack_from, mapped_list),
        's' : (mapped_bytes.pack_into, mapped_bytes.unpack_from, mapped_bytes),
        'u' : (mapped_unicode.pack_into, mapped_unicode.unpack_from, mapped_unicode),
    }
    
    del p

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if not isinstance(obj, cls):
            obj = cls(obj)
        typecode = obj.typecode
        endp = offs
        if typecode in cls.PACKERS:
            packer, padding = cls.PACKERS[typecode]
            packer.pack_into(buf, offs, typecode, obj.value)
            endp += packer.size + padding
        elif typecode in cls.OBJ_PACKERS:
            cpacker, cpadding = cls.CODE_PACKER
            cpacker.pack_into(buf, offs, typecode)
            endp += cpacker.size + cpadding
            packer = cls.OBJ_PACKERS[typecode][0]
            endp = packer(obj.value, buf, endp, idmap, implicit_offs)
        else:
            raise TypeError("Unsupported type")
        return endp

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        cpacker, cpadding = cls.CODE_PACKER
        typecode, = cpacker.unpack_from(buf, offs)
        if typecode in cls.PACKERS:
            packer, padding = cls.PACKERS[typecode]
            typecode, value = packer.unpack_from(buf, offs)
            return value
        elif typecode in cls.OBJ_PACKERS:
            offs += cpacker.size + cpadding
            unpacker = cls.OBJ_PACKERS[typecode][1]
            return unpacker(buf, offs, idmap)
        else:
            raise ValueError("Inconsistent data")

    @classmethod
    def register_schema(cls, typ, schema, typecode):
        if typ in cls.TYPE_CODES:
            if cls.TYPE_CODES[typ] != typecode or cls.OBJ_PACKERS[typecode][2].schema is not schema:
                raise ValueError("Registering different types with same typecode %r: %r" % (cls.TYPE_CODES[typ], typ))
            return cls.OBJ_PACKERS[typecode][2]

        packable = mapped_object_with_schema(schema)
        class SchemaBufferProxyProperty(GenericBufferProxyProperty):
            typ = packable
        cls.TYPE_CODES[typ] = typecode
        cls.TYPE_CODES[packable] = typecode
        cls.OBJ_PACKERS[typecode] = (packable.pack_into, packable.unpack_from, packable)
        TYPES[typ] = packable
        TYPES[packable] = packable
        PROXY_TYPES[typ] = SchemaBufferProxyProperty
        PROXY_TYPES[packable] = SchemaBufferProxyProperty
        return packable

    def __init__(self, value = None, typ = None, TYPE_CODES = TYPE_CODES):
        if value is None:
            self.typecode = None
            self.value = None
        else:
            if typ is None:
                typ = type(value)
            typ = TYPES.get(typ, typ)
            self.typecode = TYPE_CODES[typ]
            self.value = value
mapped_object.TYPE_CODES[mapped_object] = 'o'
mapped_object.OBJ_PACKERS['o'] = mapped_object.pack_into

VARIABLE_TYPES = {
    frozenset : mapped_frozenset,
    tuple : mapped_tuple,
    list : mapped_list,
    str : mapped_bytes,
    unicode : mapped_unicode,
    bytes : mapped_bytes,
    object : mapped_object,
}

FIXED_TYPES = {
    ubyte : 'B',
    uint8 : 'B',
    byte : 'b',
    int8 : 'b',
    short : 'h',
    int16 : 'h',
    ushort : 'H',
    uint16 : 'H',
    uint32 : 'I',
    int32 : 'i',
    uint64 : 'Q',
    int64 : 'q',
    int : 'q',
    long : 'q',
    float : 'd',
    float32 : 'f',
    float64 : 'd',
    bool : '?',
}

t = None
TYPES = {
    t : t
    for t in FIXED_TYPES
}
TYPES.update(VARIABLE_TYPES)
TYPES.update({
    v : v
    for v in VARIABLE_TYPES.itervalues()
})
del t

@cython.cclass
class BufferProxyObject:
    cython.declare(
        buf = object, 
        idmap = object,
        pybuf = 'Py_buffer',
        offs = cython.ulonglong, 
        none_bitmap = cython.ulonglong
    )

    @cython.locals(offs = cython.ulonglong, none_bitmap = cython.ulonglong)
    def __init__(self, buf, offs, none_bitmap, idmap = None):
        if cython.compiled:
            self.pybuf.buf = cython.NULL
        self._init(buf, offs, none_bitmap, idmap)

    @cython.ccall
    @cython.locals(offs = cython.ulonglong, none_bitmap = cython.ulonglong)
    def _init(self, buf, offs, none_bitmap, idmap):
        if cython.compiled:
            if self.pybuf.buf == cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok

        self.buf = buf
        self.idmap = idmap
        self.offs = offs
        self.none_bitmap = none_bitmap

        if cython.compiled:
            PyObject_GetBuffer(buf, cython.address(self.pybuf), PyBUF_SIMPLE)  # lint:ok

    @cython.ccall
    @cython.locals(offs = cython.ulonglong, none_bitmap = cython.ulonglong)
    def _reset(self, offs, none_bitmap, idmap):
        self.offs = offs
        self.none_bitmap = none_bitmap
        self.idmap = idmap

    if cython.compiled:
        def __del__(self):
            if self.buf is not None:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok
                self.buf = None

@cython.cclass
class BaseBufferProxyProperty:
    cython.declare(offs = cython.ulonglong, mask = cython.ulonglong)
    
    def __init__(self, offs, mask):
        self.offs = offs
        self.mask = mask

    def __set__(self, obj, value):
        raise AttributeError("Proxy objects are read-only")

    def __delete__(self, obj):
        raise AttributeError("Proxy objects are read-only")

@cython.cclass
class BoolBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.uchar)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.bint, cython.cast(cython.p_uchar,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0])  # lint:ok
        else:
            return struct.unpack_from('B', obj.buf, obj.offs + self.offs)

@cython.cclass
class UByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.uchar)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_uchar,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('B', obj.buf, obj.offs + self.offs)

@cython.cclass
class ByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.char)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.char)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_char,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('b', obj.buf, obj.offs + self.offs)

@cython.cclass
class UShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.ushort)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.ushort)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_ushort,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('H', obj.buf, obj.offs + self.offs)

@cython.cclass
class ShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.short)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.short)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_short,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('h', obj.buf, obj.offs + self.offs)

@cython.cclass
class UIntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uint)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.uint)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_uint,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('I', obj.buf, obj.offs + self.offs)

@cython.cclass
class IntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.int)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.int)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_int,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('i', obj.buf, obj.offs + self.offs)

@cython.cclass
class ULongBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.ulonglong)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.ulong)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_ulonglong,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('Q', obj.buf, obj.offs + self.offs)

@cython.cclass
class LongBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.long)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_longlong,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('q', obj.buf, obj.offs + self.offs)

@cython.cclass
class DoubleBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.double)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.double)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_double,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('d', obj.buf, obj.offs + self.offs)

@cython.cclass
class FloatBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.float)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            assert (obj.offs + self.offs + cython.sizeof(cython.float)) <= obj.pybuf.len  # lint:ok
            return cython.cast(cython.p_float,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
        else:
            return struct.unpack_from('f', obj.buf, obj.offs + self.offs)

@cython.cclass
class BytesBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong)

    @cython.locals(obj = BufferProxyObject, offs = cython.ulonglong, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
        poffs = object)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            pybuf = cython.address(obj.pybuf)
            buflen = pybuf.len
            assert (obj.offs + self.offs + cython.sizeof(cython.longlong)) <= buflen
            offs = obj.offs + cython.cast(cython.p_longlong,
                cython.cast(cython.p_uchar, pybuf.buf) + obj.offs + self.offs)[0]
            assert offs + cython.sizeof(cython.ushort) <= buflen
            if obj.idmap is not None:
                poffs = offs # python version of offs
                rv = obj.idmap.get(poffs)
                if rv is not None:
                    return rv
            rv = _unpack_bytes_from_cbuffer(cython.cast(cython.p_char, pybuf.buf), offs, buflen, None)
        else:
            offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)
            if obj.idmap is not None:
                poffs = offs
                if poffs in obj.idmap:
                    return obj.idmap[poffs]
            rv = mapped_bytes.unpack_from(obj.buf, offs)
        if obj.idmap is not None:
            obj.idmap[poffs] = rv
        return rv

@cython.cclass
class UnicodeBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong)

    @cython.locals(obj = BufferProxyObject, offs = cython.ulonglong, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
        poffs = object)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            pybuf = cython.address(obj.pybuf)
            buflen = pybuf.len
            assert (obj.offs + self.offs + cython.sizeof(cython.longlong)) <= buflen
            offs = obj.offs + cython.cast(cython.p_longlong,
                cython.cast(cython.p_uchar, pybuf.buf) + obj.offs + self.offs)[0]
            assert offs + cython.sizeof(cython.ushort) <= buflen
            if obj.idmap is not None:
                poffs = offs # python version of offs
                rv = obj.idmap.get(poffs)
                if rv is not None:
                    return rv
            rv = _unpack_bytes_from_cbuffer(cython.cast(cython.p_char, pybuf.buf), offs, buflen, None).decode("utf8")
        else:
            offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)
            if obj.idmap is not None:
                poffs = offs # python version of offs
                rv = obj.idmap.get(poffs)
                if rv is not None:
                    return rv
            rv = mapped_unicode.unpack_from(obj.buf, offs)
        if obj.idmap is not None:
            obj.idmap[poffs] = rv
        return rv

@cython.cclass
class MissingBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong)

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        else:
            raise AttributeError

@cython.cclass
class GenericBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong)

    @cython.locals(obj = BufferProxyObject, offs = cython.ulonglong, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
        poffs = object)
    def __get__(self, obj, klass):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        if cython.compiled:
            pybuf = cython.address(obj.pybuf)
            buflen = pybuf.len
            assert (obj.offs + self.offs + cython.sizeof(cython.longlong)) <= buflen
            offs = obj.offs + cython.cast(cython.p_longlong,
                cython.cast(cython.p_uchar, pybuf.buf) + obj.offs + self.offs)[0]
            if obj.idmap is not None:
                poffs = offs # python version of offs
                rv = obj.idmap.get(poffs, poffs) # idmap cannot possibly hold "poffs" for that offset
                if rv is not poffs:
                    return rv
            assert offs + cython.sizeof(cython.ushort) <= buflen
        else:
            poffs = offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)
        rv = self.typ.unpack_from(obj.buf, offs)
        if obj.idmap is not None:
            obj.idmap[poffs] = rv
        return rv

@cython.cclass
class FrozensetBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_frozenset

@cython.cclass
class TupleBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_tuple

@cython.cclass
class ListBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_list

@cython.cclass
class ObjectBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_object

PROXY_TYPES = {
    uint8 : UByteBufferProxyProperty,
    int8 : ByteBufferProxyProperty,
    uint16 : UShortBufferProxyProperty,
    int16 : ShortBufferProxyProperty,
    uint32 : UIntBufferProxyProperty,
    int32 : IntBufferProxyProperty,
    uint64 : ULongBufferProxyProperty,
    int64 : LongBufferProxyProperty,
    float64 : DoubleBufferProxyProperty,
    float32 : FloatBufferProxyProperty,
    bool : BoolBufferProxyProperty,

    mapped_frozenset : FrozensetBufferProxyProperty,
    mapped_tuple : TupleBufferProxyProperty,
    mapped_list : ListBufferProxyProperty,
    mapped_bytes : BytesBufferProxyProperty,
    mapped_unicode : UnicodeBufferProxyProperty,
    mapped_bytes : BytesBufferProxyProperty,
    mapped_object : ObjectBufferProxyProperty,
    
    int : LongBufferProxyProperty,
    float : DoubleBufferProxyProperty,
    str : BytesBufferProxyProperty,
    unicode : UnicodeBufferProxyProperty,
}

def GenericProxyClass(slot_keys, slot_types, present_bitmap, base_offs):
    class GenericProxyClass(BufferProxyObject):
        i = typ = slot = None
        value_offs = base_offs
        for i,slot in enumerate(slot_keys):
            if present_bitmap & (1 << i):
                typ = slot_types[slot]
                typ = TYPES.get(typ,typ)
                typ = PROXY_TYPES[typ]
                locals()[slot] = typ(value_offs, 1 << i)
                value_offs += typ.stride
            else:
                locals()[slot] = MissingBufferProxyProperty(0, 1 << i)
        del i, value_offs, slot, typ

    return GenericProxyClass

@cython.cclass
class Schema(object):
    cython.declare(
        slot_types = dict,
        pack_buffer_size = int,
        alignment = int,
        slot_count = int,
        slot_keys = tuple,
        slot_struct_types = dict,
        bitmap_type = str,
        bitmap_packer = object,
        bitmap_size = cython.size_t,
        packer_cache = object,
        unpacker_cache = object,
        
        _Proxy = object,
        
        _pack_buffer = object,
        _var_bitmap = cython.ulonglong,
        _fixed_bitmap = cython.ulonglong,
        _last_unpacker = tuple,
        _last_unpacker_bitmap = cython.ulonglong,
    )

    @property
    def Proxy(self):
        return functools.partial(self._Proxy, "\x00" * self.bitmap_size, 0, 0, None)
    
    def __init__(self, slot_types, alignment = 8, pack_buffer_size = 65536, packer_cache = None, unpacker_cache = None):
        if packer_cache is None:
            packer_cache = Cache(256)
        if unpacker_cache is None:
            unpacker_cache = Cache(256)
        self.slot_types = self._map_types(slot_types)
        self.pack_buffer_size = pack_buffer_size
        self.alignment = alignment
        self.packer_cache = packer_cache
        self.unpacker_cache = unpacker_cache
        self.init()

    @staticmethod
    def _map_types(slot_types):
        return { k:TYPES.get(v,v) for k,v in slot_types.iteritems() }

    @classmethod
    def from_typed_slots(cls, struct_class_or_slot_types, *p, **kw):
        if hasattr(struct_class_or_slot_types, '__slot_types__'):
            return cls(struct_class_or_slot_types.__slot_types__, *p, **kw)
        elif isinstance(struct_class_or_slot_types, dict):
            return cls(struct_class_or_slot_types, *p, **kw)
        else:
            raise ValueError("Cant build a schema out of %r" % (type(struct_class_or_slot_types),))

    def set_pack_buffer_size(self, newsize):
        self.pack_buffer_size = newsize
        self._pack_buffer = bytearray(self.pack_buffer_size)

    def init(self):
        # Freeze slot order, sort by descending size to optimize alignment
        self.slot_keys = tuple(
            sorted(
                self.slot_types.keys(),
                key = lambda k, sget = self.slot_types.get, fget = FIXED_TYPES.get : 
                    -struct.Struct(fget(sget(k), 'q')).size
            )
        )
        self.slot_count = len(self.slot_keys)
        self._pack_buffer = bytearray(self.pack_buffer_size)
        
        # Fixed types are stored inline, variable types are stored as offsets
        # into dynamically allocated buffer space not necessarily after the
        # struct's base offset (ie: for shared objects, it may point backwards
        # with negative offsets)
        self.slot_struct_types = {
            slot : FIXED_TYPES.get(typ, 'q')
            for slot, typ in self.slot_types.iteritems()
        }

        var_bitmap = 0
        fixed_bitmap = 0
        for i,slot in enumerate(self.slot_keys):
            if self.slot_types[slot] in FIXED_TYPES:
                fixed_bitmap |= 1 << i
            else:
                var_bitmap |= 1 << i
        self._var_bitmap = var_bitmap
        self._fixed_bitmap = fixed_bitmap
        self._last_unpacker = None

        if len(self.slot_keys) <= 8:
            self.bitmap_type = 'B'
        elif len(self.slot_keys) <= 16:
            self.bitmap_type = 'H'
        elif len(self.slot_keys) <= 32:
            self.bitmap_type = 'I'
        elif len(self.slot_keys) <= 64:
            self.bitmap_type = 'Q'
        else:
            raise TypeError("Too many attributes")
        self.bitmap_packer = struct.Struct("".join(["=", self.bitmap_type, self.bitmap_type]))
        self.bitmap_size = self.bitmap_packer.size

        # A proxy class for an empty object, so that users can efficiently unpack
        # into a preallocated proxy object
        self._Proxy = GenericProxyClass(self.slot_keys, self.slot_types, 0, self.bitmap_size)

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int)
    @cython.returns(tuple)
    def _get_bitmaps(self, obj):
        has_bitmap = 0
        none_bitmap = 0
        for i,slot in enumerate(self.slot_keys):
            if hasattr(obj, slot):
                has_bitmap |= 1 << i
                if getattr(obj, slot, 0) is None:
                    none_bitmap |= 1 << i
        present_bitmap = has_bitmap & ~none_bitmap
        return has_bitmap, none_bitmap, present_bitmap

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, rv = tuple)
    @cython.returns(tuple)
    def get_packer(self, obj):
        has_bitmap, none_bitmap, present_bitmap = self._get_bitmaps(obj)
        rv = self.packer_cache.get(present_bitmap)
        if rv is None:
            packer = struct.Struct("".join([
                self.bitmap_packer.format,
            ] + [
                self.slot_struct_types[slot]
                for i,slot in enumerate(self.slot_keys)
                if present_bitmap & (1 << i)
            ]))
            alignment = self.alignment
            size = packer.size
            padding = (size + alignment - 1) / alignment * alignment - size
            self.packer_cache[present_bitmap] = rv = (packer, padding)
        return rv

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, rv = tuple)
    @cython.returns(tuple)
    def get_unpacker(self, has_bitmap, none_bitmap):
        present_bitmap = has_bitmap & ~none_bitmap
        if self._last_unpacker is not None and present_bitmap == self._last_unpacker_bitmap:
            return self._last_unpacker
        rv = self.unpacker_cache.get(present_bitmap)
        if rv is None:
            pformat = "".join([
                self.slot_struct_types[slot]
                for i,slot in enumerate(self.slot_keys)
                if present_bitmap & (1 << i)
            ])
            unpacker = struct.Struct(pformat)
            alignment = self.alignment
            size = unpacker.size
            padding = (size + self.bitmap_size + alignment - 1) / alignment * alignment - size
            gfactory = GenericProxyClass(self.slot_keys, self.slot_types, present_bitmap, self.bitmap_size)
            rv = (unpacker, padding, pformat, gfactory)
            self.unpacker_cache[present_bitmap] = rv
        self._last_unpacker_bitmap = present_bitmap
        self._last_unpacker = rv
        return rv

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, alignment = int, padding = int, mask = cython.ulonglong, 
        offs = cython.longlong, implicit_offs = cython.longlong, ival_offs = cython.longlong)
    @cython.returns(tuple)
    def get_packable(self, packer, padding, obj, offs = 0, buf = None, idmap = None, implicit_offs = 0):
        if idmap is None:
            idmap = {}
        baseoffs = offs
        if buf is None:
            buf = self._pack_buffer
        has_bitmap, none_bitmap, present_bitmap = self._get_bitmaps(obj)
        fixed_present = present_bitmap & self._fixed_bitmap
        size = packer.size
        offs += size + padding
        if offs > len(buf):
            raise struct.error('buffer too small')
        packable = [
            has_bitmap,
            none_bitmap,
        ]
        packable_append = packable.append
        idmap_get = idmap.get
        slot_types = self.slot_types
        alignment = self.alignment
        for i,slot in enumerate(self.slot_keys):
            mask = 1 << i
            if present_bitmap & mask:
                val = getattr(obj, slot)
                if fixed_present & mask:
                    packable_append(val)
                else:
                    val_id = id(val)
                    val_offs = idmap_get(val_id)
                    if val_offs is None:
                        idmap[val_id] = ival_offs = offs + implicit_offs
                        offs = slot_types[slot].pack_into(val, buf, offs, idmap, implicit_offs)
                        padding = (offs + alignment - 1) / alignment * alignment - offs
                        offs += padding
                    else:
                        ival_offs = val_offs
                    packable_append(ival_offs - baseoffs - implicit_offs)

        padding = (offs + alignment - 1) / alignment * alignment - offs
        offs = offs + padding
        if offs > len(buf):
            raise struct.error('buffer too small')
        return packable, offs

    @cython.ccall
    def pack_into(self, obj, buf, offs, idmap = None, packer = None, padding = None, implicit_offs = 0):
        if idmap is None:
            idmap = {}
        if packer is None:
            packer, padding = self.get_packer(obj)
        baseoffs = offs
        packable, offs = self.get_packable(packer, padding, obj, offs, buf, idmap, implicit_offs)
        packer.pack_into(buf, baseoffs, *packable)
        if offs > len(buf):
            raise RuntimeError("Buffer overflow")
        return offs

    @cython.ccall
    def pack(self, obj, idmap = None, packer = None, padding = None, implicit_offs = 0):
        for i in xrange(24):
            try:
                endp = self.pack_into(obj, self._pack_buffer, 0, idmap, packer, padding, implicit_offs)
                return self._pack_buffer[:endp]
            except (struct.error, IndexError):
                # Buffer overflow, retry with a bigger buffer
                # Idmap is probably corrupted beyond hope though :(
                self._pack_buffer.extend(self._pack_buffer)
                if idmap is not None:
                    idmap.clear()

    @cython.ccall
    @cython.locals(
        offs = cython.longlong, padding = cython.longlong, baseoffs = cython.longlong, i = int, value_ix = int,
        has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, values = tuple, stride = cython.longlong,
        proxy_into = BufferProxyObject,
        pbuf = 'char *', pbuf2 = 'char *', pformat = 'char *', formatchar = 'char', pybuf='Py_buffer')
    def unpack_from(self, buf, offs = 0, idmap = None, factory_class_new = None, proxy_into = None):
        baseoffs = offs

        if cython.compiled:
            # Inlined bitmap unpacking
            rbuf = _likebuffer(buf)
            PyObject_GetBuffer(rbuf, cython.address(pybuf), PyBUF_SIMPLE)  # lint:ok
            assert (offs + self.bitmap_size) <= pybuf.len  # lint:ok
            pbuf = cython.cast(cython.p_char, pybuf.buf) + offs  # lint:ok

        try:
            if cython.compiled:
                if self.bitmap_size == 2:
                    has_bitmap = cython.cast(cython.p_uchar, pbuf)[0]
                    none_bitmap = cython.cast(cython.p_uchar, pbuf)[1]
                elif self.bitmap_size == 4:
                    has_bitmap = cython.cast(cython.p_ushort, pbuf)[0]
                    none_bitmap = cython.cast(cython.p_ushort, pbuf)[1]
                elif self.bitmap_size == 8:
                    has_bitmap = cython.cast(cython.p_uint, pbuf)[0]
                    none_bitmap = cython.cast(cython.p_uint, pbuf)[1]
                elif self.bitmap_size == 16:
                    has_bitmap = cython.cast(cython.p_ulonglong, pbuf)[0]
                    none_bitmap = cython.cast(cython.p_ulonglong, pbuf)[1]
                else:
                    has_bitmap, none_bitmap = self.bitmap_packer.unpack_from(buf, offs)
            else:
                has_bitmap, none_bitmap = self.bitmap_packer.unpack_from(buf, offs)
            
            if idmap is None:
                idmap = {}

            unpacker_info = self.get_unpacker(has_bitmap, none_bitmap)

            if factory_class_new is None:
                gfactory = unpacker_info[3]
                if proxy_into is None:
                    rv = gfactory(buf, offs, none_bitmap, idmap)
                else:
                    if not isinstance(proxy_into, gfactory):
                        proxy_into.__class__ = gfactory
                    if proxy_into.buf is buf:
                        proxy_into._reset(offs, none_bitmap, idmap)
                    else:
                        proxy_into._init(buf, offs, none_bitmap, idmap)
                    rv = proxy_into
            else:
                unpacker, padding, opformat, gfactory = unpacker_info
                rv = factory_class_new()

                offs += self.bitmap_size
                pbuf += self.bitmap_size
                stride = cython.cast(cython.longlong, self.bitmap_size) + padding
                
                fixed_bitmap = self._fixed_bitmap
                var_bitmap = self._var_bitmap

                if cython.compiled:
                    pformat = opformat
                    value_ix = 0
                    pbuf2 = pbuf
                    for i in xrange(self.slot_count):
                        mask = 1 << i
                        if has_bitmap & mask:
                            slot = self.slot_keys[i]
    
                            if none_bitmap & mask:
                                setattr(rv, slot, None)
                            elif fixed_bitmap & mask:
                                formatchar = pformat[value_ix]
                                if formatchar == 'B':
                                    value = cython.cast(cython.p_uchar, pbuf)[0]
                                    pbuf += cython.sizeof(cython.uchar)
                                elif formatchar == 'b':
                                    value = cython.cast(cython.p_char, pbuf)[0]
                                    pbuf += cython.sizeof(cython.char)
                                elif formatchar == '?':
                                    value = cython.cast(cython.bint, cython.cast(cython.p_char, pbuf)[0])
                                    pbuf += cython.sizeof(cython.char)
                                elif formatchar == 'H':
                                    value = cython.cast(cython.p_ushort, pbuf)[0]
                                    pbuf += cython.sizeof(cython.ushort)
                                elif formatchar == 'h':
                                    value = cython.cast(cython.p_short, pbuf)[0]
                                    pbuf += cython.sizeof(cython.short)
                                elif formatchar == 'I':
                                    value = cython.cast(cython.p_uint, pbuf)[0]
                                    pbuf += cython.sizeof(cython.uint)
                                elif formatchar == 'i':
                                    value = cython.cast(cython.p_int, pbuf)[0]
                                    pbuf += cython.sizeof(cython.int)
                                elif formatchar == 'Q':
                                    value = cython.cast(cython.p_ulonglong, pbuf)[0]
                                    pbuf += cython.sizeof(cython.ulonglong)
                                elif formatchar == 'q':
                                    value = cython.cast(cython.p_longlong, pbuf)[0]
                                    pbuf += cython.sizeof(cython.longlong)
                                elif formatchar == 'd':
                                    value = cython.cast(cython.p_double, pbuf)[0]
                                    pbuf += cython.sizeof(cython.double)
                                elif formatchar == 'f':
                                    value = cython.cast(cython.p_float, pbuf)[0]
                                    pbuf += cython.sizeof(cython.float)
                                else:
                                    raise ValueError("Inconsistent data (unknown format code %r)" % (
                                        chr(formatchar),))
                                setattr(rv, slot, value)
                                value_ix += 1
                            elif var_bitmap & mask:
                                formatchar = pformat[value_ix]
                                if formatchar == 'q':
                                    value_offs = cython.cast(cython.p_longlong, pbuf)[0] + baseoffs
                                    pbuf += cython.sizeof(cython.longlong)
                                else:
                                    raise ValueError("Inconsistent data (unexpected format code %r)" % (
                                        chr(formatchar),))
                                value_ix += 1
                                if value_offs not in idmap:
                                    slot_type = self.slot_types.get(slot)
                                    if slot_type is _mapped_bytes:
                                        value = _unpack_bytes_from_cbuffer(
                                            cython.cast(cython.p_char, pybuf.buf),   # lint:ok
                                            value_offs, pybuf.len, idmap)  # lint:ok
                                    else:
                                        value = slot_type.unpack_from(buf, value_offs, idmap)
                                    idmap[value_offs] = value
                                else:
                                    value = idmap[value_offs]
                                setattr(rv, slot, value)
                            else:
                                raise ValueError("Inconsistent data")
                    pbuf = pbuf2
                    offs += stride
                    pbuf += stride
                else:
                    values = unpacker.unpack_from(buf, offs)
                    offs += stride
        
                    value_ix = 0
                    for i in xrange(self.slot_count):
                        mask = 1 << i
                        if has_bitmap & mask:
                            slot = self.slot_keys[i]
                            if none_bitmap & mask:
                                setattr(rv, slot, None)
                            elif fixed_bitmap & mask:
                                setattr(rv, slot, values[value_ix])
                                value_ix += 1
                            elif var_bitmap & mask:
                                value_offs = cython.cast(cython.longlong, values[value_ix]) + baseoffs
                                value_ix += 1
                                if value_offs not in idmap:
                                    slot_type = self.slot_types.get(slot)
                                    idmap[value_offs] = value = slot_type.unpack_from(buf, value_offs, idmap)
                                else:
                                    value = idmap[value_offs]
                                setattr(rv, slot, value)
                            else:
                                raise ValueError("Inconsistent data")
            return rv
        finally:
            if cython.compiled:
                PyBuffer_Release(cython.address(pybuf))  # lint:ok

    def unpack(self, buf, idmap = None, factory_class_new = None, proxy_into = None):
        return self.unpack_from(buffer(buf), 0, idmap, factory_class_new, proxy_into)

    def __getstate__(self):
        return (self.slot_types, self.alignment, self.pack_buffer_size)

    def __setstate__(self, state):
        self.slot_types, self.alignment, self.pack_buffer_size = state
        self.init()


class mapped_object_with_schema(object):
    schema = cython.declare(Schema)
    
    def __init__(self, schema):
        self.schema = schema
        
    def pack_into(self, obj, buf, offs, idmap = None, implicit_offs = 0):
        return self.schema.pack_into(obj, buf, offs, idmap, implicit_offs = implicit_offs)

    def unpack_from(self, buf, offs, idmap = None):
        return self.schema.unpack_from(buf, offs, idmap)

class MappedArrayProxyBase(object):
    # Must subclass to select a schema and proxy class
    schema = None
    proxy_class = None

    _Header = struct.Struct("=QQQ")
    
    def __init__(self, buf, offset = 0, idmap = None, idmap_size = 1024):
        if idmap is None:
            idmap = Cache(idmap_size)
        self.offset = offset
        if offset != 0:
            self.buf = buf = buffer(buf, offset)
        else:
            self.buf = buf
        self.total_size, self.index_offset, self.index_elements = self._Header.unpack_from(buf, 0)
        self.index = numpy.frombuffer(buf, 
            offset = self.index_offset, 
            dtype = numpy.uint64, 
            count = self.index_elements)
        self.idmap = idmap

    def __getitem__(self, pos):
        return self.getter()(pos)

    @cython.locals(schema = Schema, proxy_into = BufferProxyObject)
    def getter(self, proxy_into = None, no_idmap = False):
        schema = self.schema
        proxy_class = self.proxy_class
        index = self.index
        idmap = self.idmap if not no_idmap else None
        buf = self.buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None
        
        @cython.locals(pos=int)
        def getter(pos):
            return schema.unpack_from(buf, index[pos], idmap, proxy_class_new, proxy_into)
        return getter

    @cython.locals(i = int, schema = Schema)
    def __iter__(self):
        # getter inlined
        schema = self.schema
        proxy_class = self.proxy_class
        index = self.index
        idmap = self.idmap
        buf = self.buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None

        for i in xrange(len(self)):
            yield schema.unpack_from(buf, index[i], idmap, proxy_class_new)

    @cython.locals(i = int, schema = Schema)
    def iter_fast(self):
        # getter inlined
        schema = self.schema
        proxy_class = self.proxy_class
        index = self.index
        idmap = self.idmap
        buf = self.buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None

        proxy_into = schema.Proxy()
        for i in xrange(len(self)):
            yield schema.unpack_from(buf, index[i], idmap, proxy_class_new, proxy_into)

    def __len__(self):
        return len(self.index)

    @classmethod
    @cython.locals(schema = Schema, data_pos = cython.size_t, initial_pos = cython.size_t, current_pos = object)
    def build(cls, initializer, destfile = None, tempdir = None, idmap = None):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        initial_pos = destfile.tell()
        write = destfile.write
        write(cls._Header.pack(0, 0, 0))
        destfile.flush()
        data_pos = destfile.tell()
        schema = cls.schema
        _index = []
        index_parts = []
        for item in initializer:
            current_pos = data_pos - initial_pos
            _index.append(current_pos)
            buf = schema.pack(item, idmap, None, None, current_pos)
            write(buf)
            data_pos += len(buf)
            if len(_index) >= 100000:
                index_parts.append(numpy.array(_index, dtype = numpy.uint64))
                del _index[:]
        destfile.flush()
        index_pos = destfile.tell()
        if _index:
            index_parts.append(numpy.array(_index, dtype = numpy.uint64))
            del _index
        if index_parts:
            index = numpy.concatenate(numpy.array(index_parts))
        else:
            index = numpy.array([], dtype = numpy.uint64)
        del index_parts
        index = numpy.array(index, dtype = numpy.uint64)
        write(buffer(index))
        destfile.flush()
        final_pos = destfile.tell()
        destfile.seek(initial_pos)
        write(cls._Header.pack(final_pos - initial_pos, index_pos - initial_pos, len(index)))
        destfile.flush()
        destfile.seek(final_pos)
        return cls.map_file(destfile, initial_pos)

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None):
        fileobj.seek(offset)
        total_size = cls._Header.unpack(fileobj.read(cls._Header.size))[0]
        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        buf = mmap.mmap(fileobj.fileno(), total_size + offset - map_start, 
            access = mmap.ACCESS_READ, offset = map_start)
        rv = cls(buffer(buf, offset - map_start))
        rv._file = fileobj
        rv._mmap = buf
        return rv

if cython.compiled:
    @cython.cfunc
    @cython.locals(
        hkey = cython.ulonglong,
        mkey = cython.ulonglong,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_ui64(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        if lo < hi:
            # First iteration a quick guess assuming uniform distribution of keys
            mid = min(hint, hi-1)
            mkey = cython.cast(cython.p_ulonglong, pindex + stride0 * mid)[0]
            if mkey < hkey:
                # Got a lo guess, now skip-search forward for a hi
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_ulonglong, pindex + stride0 * (mid+skip))[0] < hkey:
                        lo = mid+1
                        mid += skip
                        skip *= 2
                    else:
                        hi = mid + skip
                        break
            elif mkey > hkey:
                # Got a hi guess, now skip-search backwards for a lo
                hi = mid
                skip = 32
                while skip > 0 and mid > lo + skip:
                    if cython.cast(cython.p_ulonglong, pindex + stride0 * (mid-skip))[0] > hkey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                # hit, but must find the first
                # good idea to go sequential because we assume collisions are unlikely
                while mid > lo and cython.cast(cython.p_ulonglong, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        # Final stretch: search the remaining range with a regular binary search
        while lo < hi:
            mid = (lo+hi)//2
            mkey = cython.cast(cython.p_ulonglong, pindex + stride0 * mid)[0]
            if mkey < hkey:
                lo = mid+1
            elif mkey > hkey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_ulonglong, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        return lo

    @cython.cfunc
    @cython.locals(
        hkey = cython.ulonglong,
        mkey = cython.ulonglong, uikey = cython.uint, uimkey = cython.uint,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_ui32(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        uikey = hkey
        if lo < hi:
            mid = min(hint, hi-1)
            uimkey = cython.cast(cython.p_uint, pindex + stride0 * mid)[0]
            if uimkey < uikey:
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_uint, pindex + stride0 * (mid+skip))[0] < uikey:
                        lo = mid+1
                        mid += skip
                        skip *= 2
                    else:
                        hi = mid + skip
                        break
            elif uimkey > uikey:
                hi = mid
                skip = 32
                while skip > 0 and mid > lo + skip:
                    if cython.cast(cython.p_uint, pindex + stride0 * (mid-skip))[0] > uikey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                while mid > lo and cython.cast(cython.p_uint, pindex + stride0 * (mid-1))[0] == uikey:
                    mid -= 1
                return mid
        while lo < hi:
            mid = (lo+hi)//2
            uimkey = cython.cast(cython.p_uint, pindex + stride0 * mid)[0]
            if uimkey < uikey:
                lo = mid+1
            elif uimkey > uikey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_uint, pindex + stride0 * (mid-1))[0] == uikey:
                    mid -= 1
                return mid
        return lo

@cython.cclass
class NumericIdMapper(object):
    dtype = npuint64

    # Num-items, Index-pos
    _Header = struct.Struct('=QQ')

    cython.declare(
        _buf = object,
        _likebuf = object,
        _file = object,
        _dtype = object,
        index_elements = cython.ulonglong,
        index_offset = cython.ulonglong,
        index = object,
        _index_min = cython.ulonglong,
        _index_max = cython.ulonglong,
        dtypemax = cython.ulonglong,
    )

    @property
    def buf(self):
        return self._buf

    def __init__(self, buf, offset = 0):
        # Accelerate class attributes
        self._dtype = self.dtype

        # Initialize buffer
        if offset:
            self._buf = self._likebuf = buffer(buf, offset)
        else:
            self._buf = buf
            self._likebuf = _likebuffer(buf)

        # Parse header and map index
        self.index_elements, self.index_offset = self._Header.unpack_from(self._buf, 0)

        self.index = numpy.ndarray(buffer = self._buf, 
            offset = self.index_offset, 
            dtype = self.dtype, 
            shape = (self.index_elements, 2))

        if len(self.index) > 0:
            self._index_min = self.index[0,0]
            self._index_max = self.index[-1,0]
        else:
            self._index_min = self._index_max = 0

        dtype = self.dtype
        try:
            self.dtypemax = ~dtype(0)
        except:
            self.dtypemax = ~0

    def __getitem__(self, key):
        rv = self.get(key)
        if rv is None:
            raise KeyError(key)
        else:
            return rv

    def __contains__(self, key):
        return self.get(key) is not None

    def preload(self):
        # Just touch everything in sequential order
        self.index.max()

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer', pybuf = 'Py_buffer')
    def iterkeys(self):
        return iter(self.index[:,0])

    def keys(self):
        # Baaad idea
        return self.index[:,0]

    def __iter__(self):
        return self.iterkeys()

    def itervalues(self):
        return iter(self.index[:,1])

    def values(self):
        return self.index[:,1]

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer', pybuf = 'Py_buffer',
        stride0 = cython.size_t, stride1 = cython.size_t, pindex = cython.p_char)
    def iteritems(self):
        buf = self._buf
        dtype = self.dtype
        index = self.index
        if cython.compiled:
            #lint:disable
            buf = self._likebuf
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
            try:
                if dtype is npuint64:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                    try:
                        if ( indexbuf.strides == cython.NULL 
                                or indexbuf.ndim < 2
                                or indexbuf.len < self.index_elements * indexbuf.strides[0] ):
                            raise ValueError("Invalid buffer state")
                        stride0 = indexbuf.strides[0]
                        stride1 = indexbuf.strides[1]
                        pindex = cython.cast(cython.p_char, indexbuf.buf)
                        for i in xrange(self.index_elements):
                            yield (
                                cython.cast(cython.p_ulonglong, pindex)[0],
                                cython.cast(cython.p_ulonglong, pindex + stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                elif dtype is npuint32:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                    try:
                        if ( indexbuf.strides == cython.NULL 
                                or indexbuf.ndim < 2
                                or indexbuf.len < self.index_elements * indexbuf.strides[0] ):
                            raise ValueError("Invalid buffer state")
                        stride0 = indexbuf.strides[0]
                        stride1 = indexbuf.strides[1]
                        pindex = cython.cast(cython.p_char, indexbuf.buf)
                        for i in xrange(self.index_elements):
                            yield (
                                cython.cast(cython.p_uint, pindex)[0],
                                cython.cast(cython.p_uint, pindex + stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in xrange(self.index_elements):
                        yield (
                            index[i,0],
                            index[i,1]
                        )
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in xrange(self.index_elements):
                yield (index[i,0], index[i,1])

    def items(self):
        # Bad idea, but hey, if they do this, it means the caller expects the collection to fit in RAM
        return list(self.iteritems())

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong,
        lo = cython.size_t, hi = cython.size_t, hint = cython.size_t, stride0 = cython.size_t,
        indexbuf = 'Py_buffer', pindex = cython.p_char)
    @cython.returns(cython.size_t)
    def _search_hkey(self, hkey):
        hi = self.index_elements
        lo = 0
        hikey = self._index_max
        lokey = self._index_min
        if hkey < lokey:
            return lo
        elif hkey > hikey:
            return hi
        if cython.compiled:
            dtype = self._dtype
            if dtype is npuint64 or dtype is npuint32:
                #lint:disable
                PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                try:
                    if ( indexbuf.strides == cython.NULL 
                            or indexbuf.len < hi * indexbuf.strides[0] ):
                        raise ValueError("Invalid buffer state")
                    pindex = cython.cast(cython.p_char, indexbuf.buf)
                    stride0 = indexbuf.strides[0]

                    if dtype is npuint64:
                        # TO-DO: better hints?
                        hint = (lo+hi)//2
                        return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint)
                    elif dtype is npuint32:
                        # TO-DO: better hints?
                        hint = (lo+hi)//2
                        return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint)
                    else:
                        raise AssertionError("Internal error")
                finally:
                    PyBuffer_Release(cython.address(indexbuf))
                #lint:enable
            else:
                raise AssertionError("Internal error")
        return lo

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get(self, key, default = None):
        if not isinstance(key, (int, long)):
            return default
        if key < 0 or key > self.dtypemax:
            return default
        hkey = key
        startpos = self._search_hkey(hkey)
        nitems = self.index_elements
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                #lint:disable
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    if dtype is npuint64:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            if pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                return cython.cast(cython.p_ulonglong, pindex + stride1)[0]
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    elif dtype is npuint32:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            if pindex < pindexend and cython.cast(cython.p_uint, pindex)[0] == hkey:
                                return cython.cast(cython.p_uint, pindex + stride1)[0]
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        if startpos < nitems and index[startpos,0] == hkey:
                            return index[startpos,1]
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                if startpos < nitems and index[startpos,0] == hkey:
                    return index[startpos,1]
        return default

    @classmethod
    @cython.locals(
        basepos = cython.ulonglong, curpos = cython.ulonglong, endpos = cython.ulonglong, finalpos = cython.ulonglong,
        discard_duplicates = cython.bint, discard_duplicate_keys = cython.bint)
    def build(cls, initializer, destfile = None, tempdir = None, 
            discard_duplicates = False, discard_duplicate_keys = False):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        basepos = destfile.tell()

        # Reserve space for the header
        write = destfile.write
        write(cls._Header.pack(0, 0))

        # Build the index - the index is a matrix of the form:
        # [ [ key, value id ], ... ]
        #
        # With the rows ordered by hash
        if isinstance(initializer, dict):
            initializer = initializer.iteritems()
        else:
            initializer = iter(initializer)
        parts = []
        islice = itertools.islice
        array = numpy.array
        unique = numpy.unique
        curpos = basepos + cls._Header.size
        bytes_pack_into = mapped_bytes.pack_into
        valbuf = bytearray(65536)
        valbuflen = 65536
        n = 0
        part = []
        while 1:
            del part[:]
            for k,i in islice(initializer, 1000):
                # Add the index item
                n += 1
                part.append((k,i))
            if part:
                apart = array(part, dtype)
                if discard_duplicate_keys:
                    apart = apart[unique(apart[:,0], return_index=True)[1]]
                elif discard_duplicates:
                    apart = apart[unique(apart.view(numpy.dtype([
                            ('key', dtype),
                            ('value', dtype),
                        ])), return_index=True)[1]]
                parts.append(apart)
            else:
                break
        if parts:
            index = numpy.concatenate(array(parts))
            if discard_duplicate_keys:
                index = index[unique(index[:,0], return_index=True)[1]]
            elif discard_duplicates:
                index = index[unique(index.view(numpy.dtype([
                        ('key', dtype),
                        ('value', dtype),
                    ])), return_index=True)[1]]
            else:
                shuffle = numpy.argsort(index[:,0])
                index = index[shuffle]
                del shuffle
        else:
            index = numpy.empty(shape=(0,2), dtype=dtype)
        del parts, part

        indexpos = curpos
        write(buffer(index))
        nitems = len(index)

        finalpos = destfile.tell()
        if finalpos & 31:
            write("\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
        destfile.seek(finalpos)
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'NumericIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None):
        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        fileobj.seek(map_start)
        buf = mmap.mmap(fileobj.fileno(), 0, access = mmap.ACCESS_READ, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

class NumericId32Mapper(NumericIdMapper):
    dtype = npuint32

@cython.locals(ux = unicode)
def safe_utf8(x):
    if isinstance(x, unicode):
        # The assignment-style cast is needed because an inline case triggers a Cython compiler crash
        ux = x
        return ux.encode("utf8")
    else:
        return x

@cython.cclass
class StringIdMapper(object):
    encode = staticmethod(safe_utf8)
    dtype = npuint64
    xxh = xxhash.xxh64

    # Num-items, Index-pos
    _Header = struct.Struct('=QQ')

    cython.declare(
        _buf = object,
        _likebuf = object,
        _file = object,
        _encode = object,
        _dtype = object,
        _xxh = object,
        index_elements = cython.ulonglong,
        index_offset = cython.ulonglong,
        index = object,
    )

    @property
    def buf(self):
        return self._buf

    def __init__(self, buf, offset = 0):
        # Accelerate class attributes
        self._encode = self.encode
        self._dtype = self.dtype
        self._xxh = self.xxh

        # Initialize buffer
        if offset:
            self._buf = self._likebuf = buffer(buf, offset)
        else:
            self._buf = buf
            self._likebuf = _likebuffer(buf)

        # Parse header and map index
        self.index_elements, self.index_offset = self._Header.unpack_from(self._buf, 0)

        self.index = numpy.ndarray(buffer = self._buf, 
            offset = self.index_offset, 
            dtype = self.dtype, 
            shape = (self.index_elements, 3))

    def __getitem__(self, key):
        rv = self.get(key)
        if rv is None:
            raise KeyError(key)
        else:
            return rv

    def __contains__(self, key):
        return self.get(key) is not None

    def preload(self):
        # Just touch everything in sequential order
        self.index.max()

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer', pybuf = 'Py_buffer')
    def iterkeys(self, make_sequential = True):
        buf = self._buf
        dtype = self.dtype
        if make_sequential:
            # Big collections don't fit in RAM, so it helps if they're accessed sequentially
            # We'll just have to copy and sort the index, no big deal though
            index = numpy.sort(self.index[:,1])
            stride = 1
            offs = 0
        else:
            index = self.index
            stride = 3
            offs = 1
        if cython.compiled:
            #lint:disable
            buf = self._likebuf
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
            try:
                if dtype is npuint64:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_SIMPLE)
                    try:
                        if indexbuf.len < (self.index_elements * stride * cython.sizeof(cython.ulonglong)):
                            raise ValueError("Invalid buffer state")
                        for i in xrange(self.index_elements):
                            yield _unpack_bytes_from_cbuffer(
                                cython.cast(cython.p_char, pybuf.buf),
                                cython.cast(cython.p_ulonglong, indexbuf.buf)[i*stride+offs],
                                pybuf.len, None)
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                elif dtype is npuint32:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_SIMPLE)
                    try:
                        if indexbuf.len < (self.index_elements * stride * cython.sizeof(cython.uint)):
                            raise ValueError("Invalid buffer state")
                        for i in xrange(self.index_elements):
                            yield _unpack_bytes_from_cbuffer(
                                cython.cast(cython.p_char, pybuf.buf),
                                cython.cast(cython.p_uint, indexbuf.buf)[i*stride+offs],
                                pybuf.len, None)
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in xrange(self.index_elements):
                        yield _unpack_bytes_from_cbuffer(
                            cython.cast(cython.p_char, pybuf.buf),
                            index[i],
                            pybuf.len, None)
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in xrange(self.index_elements):
                yield _unpack_bytes_from_pybuffer(buf, index[i], None)

    def __iter__(self):
        return self.iterkeys()

    def itervalues(self):
        return iter(self.index[:,2])

    def values(self):
        return self.index[:,2]

    def keys(self):
        # Baaad idea
        return list(self.iterkeys())

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer', pybuf = 'Py_buffer',
        stride0 = cython.size_t, stride1 = cython.size_t, pindex = cython.p_char)
    def iteritems(self, make_sequential = True):
        buf = self._buf
        dtype = self.dtype
        if make_sequential:
            # Big collections don't fit in RAM, so it helps if they're accessed sequentially
            # We'll just have to copy and sort the index, no big deal though
            index = self.index[numpy.argsort(self.index[:,1])]
        else:
            index = self.index
        if cython.compiled:
            #lint:disable
            buf = self._likebuf
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
            try:
                if dtype is npuint64:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                    try:
                        if ( indexbuf.strides == cython.NULL 
                                or indexbuf.ndim < 2
                                or indexbuf.len < self.index_elements * indexbuf.strides[0] ):
                            raise ValueError("Invalid buffer state")
                        stride0 = indexbuf.strides[0]
                        stride1 = indexbuf.strides[1]
                        pindex = cython.cast(cython.p_char, indexbuf.buf)
                        for i in xrange(self.index_elements):
                            yield (
                                _unpack_bytes_from_cbuffer(
                                    cython.cast(cython.p_char, pybuf.buf),
                                    cython.cast(cython.p_ulonglong, pindex + stride1)[0],
                                    pybuf.len, None),
                                cython.cast(cython.p_ulonglong, pindex + 2*stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                elif dtype is npuint32:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                    try:
                        if ( indexbuf.strides == cython.NULL 
                                or indexbuf.ndim < 2
                                or indexbuf.len < self.index_elements * indexbuf.strides[0] ):
                            raise ValueError("Invalid buffer state")
                        stride0 = indexbuf.strides[0]
                        stride1 = indexbuf.strides[1]
                        pindex = cython.cast(cython.p_char, indexbuf.buf)
                        for i in xrange(self.index_elements):
                            yield (
                                _unpack_bytes_from_cbuffer(
                                    cython.cast(cython.p_char, pybuf.buf),
                                    cython.cast(cython.p_uint, pindex + stride1)[0],
                                    pybuf.len, None),
                                cython.cast(cython.p_uint, pindex + 2*stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in xrange(self.index_elements):
                        yield (
                            _unpack_bytes_from_cbuffer(
                                cython.cast(cython.p_char, pybuf.buf),
                                index[i,1],
                                pybuf.len, None),
                            index[i,2]
                        )
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in xrange(self.index_elements):
                yield (_unpack_bytes_from_pybuffer(buf, index[i,1], None), index[i,2])

    def items(self):
        # Bad idea, but hey, if they do this, it means the caller expects the collection to fit in RAM
        return list(self.iteritems(make_sequential = False))

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong,
        lo = cython.size_t, hi = cython.size_t, hint = cython.size_t, stride0 = cython.size_t,
        indexbuf = 'Py_buffer', pindex = cython.p_char)
    @cython.returns(cython.size_t)
    def _search_hkey(self, hkey):
        hi = self.index_elements
        lo = 0
        if cython.compiled:
            dtype = self._dtype
            if dtype is npuint64 or dtype is npuint32:
                #lint:disable
                PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                try:
                    if ( indexbuf.strides == cython.NULL 
                            or indexbuf.len < hi * indexbuf.strides[0] ):
                        raise ValueError("Invalid buffer state")
                    pindex = cython.cast(cython.p_char, indexbuf.buf)
                    stride0 = indexbuf.strides[0]

                    if dtype is npuint64:
                        # A quick guess assuming uniform distribution of keys over the 64-bit value range
                        hint = (((hkey >> 32) * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint)
                    elif dtype is npuint32:
                        # A quick guess assuming uniform distribution of keys over the 64-bit value range
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint)
                    else:
                        raise AssertionError("Internal error")
                finally:
                    PyBuffer_Release(cython.address(indexbuf))
                #lint:enable
        return lo

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get(self, key, default = None):
        if not isinstance(key, basestring):
            return default
        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()
        startpos = self._search_hkey(hkey)
        nitems = self.index_elements
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                pbkey = bkey
                blen = len(bkey)
                #lint:disable
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    if dtype is npuint64:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                if _compare_bytes_from_cbuffer(pbkey, blen,
                                        cython.cast(cython.p_char, pybuf.buf),
                                        cython.cast(cython.p_ulonglong, pindex + stride1)[0],
                                        pybuf.len):
                                    return cython.cast(cython.p_ulonglong, pindex + 2*stride1)[0]
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    elif dtype is npuint32:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_uint, pindex)[0] == hkey:
                                if _compare_bytes_from_cbuffer(pbkey, blen,
                                        cython.cast(cython.p_char, pybuf.buf),
                                        cython.cast(cython.p_uint, pindex + stride1)[0],
                                        pybuf.len):
                                    return cython.cast(cython.p_uint, pindex + 2*stride1)[0]
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            if _compare_bytes_from_cbuffer(pbkey, blen,
                                    cython.cast(cython.p_char, pybuf.buf),
                                    self.index[startpos,1],
                                    pybuf.len) == bkey:
                                return index[startpos,2]
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    if _unpack_bytes_from_pybuffer(buf, index[startpos,1], None) == bkey:
                        return index[startpos,2]
                    startpos += 1
        return default

    @classmethod
    @cython.locals(
        basepos = cython.ulonglong, curpos = cython.ulonglong, endpos = cython.ulonglong, finalpos = cython.ulonglong,
        dtypemax = cython.ulonglong)
    def build(cls, initializer, destfile = None, tempdir = None):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        try:
            dtypemax = ~dtype(0)
        except:
            dtypemax = ~0

        basepos = destfile.tell()

        # Reserve space for the header
        write = destfile.write
        write(cls._Header.pack(0, 0))

        # Build the index - the index is a matrix of the form:
        # [ [ hash, key offset, value id ], ... ]
        #
        # With the rows ordered by hash
        if isinstance(initializer, dict):
            initializer = initializer.iteritems()
        else:
            initializer = iter(initializer)
        parts = []
        xxh = cls.xxh
        islice = itertools.islice
        array = numpy.array
        curpos = basepos + cls._Header.size
        encode = cls.encode
        bytes_pack_into = mapped_bytes.pack_into
        valbuf = bytearray(65536)
        valbuflen = 65536
        n = 0
        part = []
        while 1:
            del part[:]
            for k,i in islice(initializer, 1000):
                # Add the index item
                n += 1
                k = encode(k)
                klen = len(k)
                if curpos > dtypemax:
                    raise ValueError("Cannot represent offset with requested precision")
                part.append((xxh(k).intdigest(),curpos,i))
                if klen + 16 > valbuflen:
                    valbuflen = (klen + 16) * 2
                    valbuf = bytearray(valbuflen)
                endpos = bytes_pack_into(k, valbuf, 0)
                if valbuflen < endpos:
                    raise RuntimeError("Buffer overflow")
                write(valbuf[:endpos])
                curpos += endpos
            if part:
                parts.append(array(part, dtype))
            else:
                break
        if parts:
            index = numpy.concatenate(numpy.array(parts))
        else:
            index = numpy.empty(shape=(0,3), dtype=dtype)
        shuffle = numpy.argsort(index[:,0])
        index = index[shuffle]
        del parts, part

        indexpos = curpos
        write(buffer(index))
        nitems = len(index)

        finalpos = destfile.tell()
        if finalpos & 31:
            write("\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
        destfile.seek(finalpos)
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'StringIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None):
        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        fileobj.seek(map_start)
        buf = mmap.mmap(fileobj.fileno(), 0, access = mmap.ACCESS_READ, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

@cython.cclass
class StringId32Mapper(StringIdMapper):
    dtype = npuint32
    xxh = xxhash.xxh32

@cython.cclass
class NumericIdMultiMapper(NumericIdMapper):
    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get(self, key, default = None):
        if not isinstance(key, (int, long)):
            return default
        if key < 0 or key > self.dtypemax:
            return default
        hkey = key
        startpos = self._search_hkey(hkey)
        nitems = self.index_elements
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                #lint:disable
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    rv = []
                    if dtype is npuint64:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                rv.append(cython.cast(cython.p_ulonglong, pindex + stride1)[0])
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    elif dtype is npuint32:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_uint, pindex)[0] == hkey:
                                rv.append(cython.cast(cython.p_uint, pindex + stride1)[0])
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            rv.append(index[startpos,1])
                            startpos += 1
                    if rv:
                        return rv
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                rv = []
                while startpos < nitems and index[startpos,0] == hkey:
                    rv.append(index[startpos,1])
                    startpos += 1
                if rv:
                    return rv
        return default

class NumericId32MultiMapper(NumericIdMultiMapper):
    dtype = npuint32

@cython.cclass
class StringIdMultiMapper(StringIdMapper):
    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get(self, key, default = None):
        if not isinstance(key, basestring):
            return default
        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()
        startpos = self._search_hkey(hkey)
        nitems = self.index_elements
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                pbkey = bkey
                blen = len(bkey)
                #lint:disable
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    rv = []
                    if dtype is npuint64:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                if _compare_bytes_from_cbuffer(pbkey, blen,
                                        cython.cast(cython.p_char, pybuf.buf),
                                        cython.cast(cython.p_ulonglong, pindex + stride1)[0],
                                        pybuf.len):
                                    rv.append(cython.cast(cython.p_ulonglong, pindex + 2*stride1)[0])
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    elif dtype is npuint32:
                        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                        try:
                            if ( indexbuf.strides == cython.NULL 
                                    or indexbuf.ndim < 2
                                    or indexbuf.len < nitems * indexbuf.strides[0] ):
                                raise ValueError("Invalid buffer state")
                            stride0 = indexbuf.strides[0]
                            stride1 = indexbuf.strides[1]
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            while pindex < pindexend and cython.cast(cython.p_uint, pindex)[0] == hkey:
                                if _compare_bytes_from_cbuffer(pbkey, blen,
                                        cython.cast(cython.p_char, pybuf.buf),
                                        cython.cast(cython.p_uint, pindex + stride1)[0],
                                        pybuf.len):
                                    rv.append(cython.cast(cython.p_uint, pindex + 2*stride1)[0])
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            if _compare_bytes_from_cbuffer(pbkey, blen,
                                    cython.cast(cython.p_char, pybuf.buf),
                                    self.index[startpos,1],
                                    pybuf.len) == bkey:
                                rv.append(index[startpos,2])
                            startpos += 1
                    if rv:
                        return rv
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                rv = []
                while startpos < nitems and index[startpos,0] == hkey:
                    if _unpack_bytes_from_pybuffer(buf, index[startpos,1], None) == bkey:
                        rv.append(index[startpos,2])
                    startpos += 1
                if rv:
                    return rv
        return default

class StringId32MultiMapper(StringIdMultiMapper):
    dtype = npuint32
    xxh = xxhash.xxh32

@cython.cclass
class ApproxStringIdMultiMapper(NumericIdMultiMapper):
    encode = staticmethod(safe_utf8)
    xxh = xxhash.xxh64

    cython.declare(
        _encode = object,
        _xxh = object,
    )

    def __init__(self, buf, offset = 0):
        # Accelerate class attributes
        self._encode = self.encode
        self._xxh = self.xxh
        NumericIdMultiMapper.__init__(self, buf, offset)

    @cython.ccall
    def get(self, key, default = None):
        return super(ApproxStringIdMultiMapper, self).get(self._xxh(self._encode(key)).intdigest(), default)

    @classmethod
    def build(cls, initializer, *p, **kw):
        xxh = cls.xxh
        encode = cls.encode
        def wrapped_initializer():
            for key, value in initializer:
                yield xxh(encode(key)).intdigest(), value
        return super(ApproxStringIdMultiMapper, cls).build(wrapped_initializer(), *p, **kw)

@cython.cclass
class ApproxStringId32MultiMapper(ApproxStringIdMultiMapper):
    xxh = xxhash.xxh32
    dtype = npuint32

@cython.locals(i = int)
def _iter_values_dump_keys(items, keys_file, value_cache_size = 1024):
    if isinstance(items, dict):
        items = items.iteritems()
    dump = cPickle.dump
    i = -1
    value_cache = Cache(value_cache_size)
    for key, value in items:
        if value not in value_cache:
            yield value
            value_cache[value] = value
            i += 1
        dump((key, i), keys_file, 2)
    keys_file.flush()

def _iter_key_dump(keys_file):
    keys_file.seek(0)
    load = cPickle.load
    while 1:
        try:
            yield load(keys_file)
        except EOFError:
            break

class MappedMappingProxyBase(object):
    # Must subclass to select mapping strategies

    # A MappedArrayProxyBase subclass for values
    ValueArray = None

    # A MappedIdMappingBase subclass for keys
    IdMapper = None

    _Footer = struct.Struct("=Q")
    
    def __init__(self, value_array, id_mapper):
        self.value_array = value_array
        self.id_mapper = id_mapper

    def __getitem__(self, key):
        return self.value_array[self.id_mapper[key]]

    def __contains__(self, key):
        return key in self.id_mapper

    def values(self):
        return self.value_array

    def itervalues(self):
        return iter(self.value_array)

    def iterkeys(self):
        return iter(self.id_mapper)

    def keys(self):
        return self.id_mapper.keys()

    def iteritems(self):
        g = self.value_array.getter()
        for k,i in self.id_mapper.iteritems():
            yield (k, g(i))

    def items(self):
        return list(self.iteritems())

    def __iter__(self):
        return iter(self.id_mapper)

    def get(self, key, default = None):
        ix = self.id_mapper.get(key)
        if ix is None:
            return default
        else:
            return self.value_array[ix]

    def __len__(self):
        return len(self.value_array)

    @classmethod
    def build(cls, initializer, destfile = None, tempdir = None, idmap = None,
            value_array_kwargs = {},
            id_mapper_kwargs = {}):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        # Must dump values and keys to temporary files because
        # we don't know the size of the idmap before building it,
        # and it has to be at the beginning of the file
        with tempfile.NamedTemporaryFile(dir = tempdir, suffix = '.v',) as values_file:
            with tempfile.NamedTemporaryFile(dir = tempdir, suffix = '.k') as keys_file:
                initial_pos = destfile.tell()

                value_array = cls.ValueArray.build(
                    _iter_values_dump_keys(initializer, keys_file), values_file,
                    tempdir = tempdir, idmap = idmap, **value_array_kwargs)

                id_mapper = cls.IdMapper.build(
                    _iter_key_dump(keys_file), destfile, 
                    tempdir = tempdir, **id_mapper_kwargs)

            # pad to multiple of 32 for better cache alignment
            pos = destfile.tell()
            if pos & 31:
                destfile.write("\x00" * (32 - (pos & 31)))
    
            values_pos = destfile.tell()

            blocklen = 1 << 20
            for start in xrange(0, len(value_array.buf), blocklen):
                destfile.write(buffer(value_array.buf, start, blocklen))
            destfile.write(cls._Footer.pack(values_pos - initial_pos))
            destfile.flush()

            return cls(value_array, id_mapper)

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        values_pos, = cls._Footer.unpack_from(buf, offset + len(buf) - cls._Footer.size)
        value_array = cls.ValueArray.map_buffer(buf, offset + values_pos)
        id_mapper = cls.IdMapper.map_buffer(buf, offset)
        return cls(value_array, id_mapper)

    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None):
        # If no size is given, it's the whole file by default
        if size is None:
            fileobj.seek(0, os.SEEK_END)
            size = fileobj.tell() - offset

        # Read the footer
        fileobj.seek(offset + size - cls._Footer.size)
        values_pos, = cls._Footer.unpack(fileobj.read(cls._Footer.size))
        fileobj.seek(offset)

        # Map everything
        id_mapper = cls.IdMapper.map_file(fileobj, offset, size = values_pos)
        value_array = cls.ValueArray.map_file(fileobj, offset + values_pos, 
            size = size - cls._Footer.size - values_pos)
        return cls(value_array, id_mapper)


class MappedMultiMappingProxyBase(MappedMappingProxyBase):
    def __getitem__(self, key):
        ids = self.id_mapper[key]
        return [ self.value_array[id_] for id_ in ids ]

    def get(self, key, default = None):
        ixs = self.id_mapper.get(key)
        if ixs is None:
            return default
        else:
            return [ self.value_array[ix] for ix in ixs ]
