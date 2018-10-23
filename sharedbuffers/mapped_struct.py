# -*- coding: utf-8 -*-
# cython: infer_types=True, profile=False, linetrace=False
# distutils: define_macros=CYTHON_TRACE=0
import struct
import array
import mmap
import numpy
import tempfile
import functools
import cPickle
import os
import sys
import xxhash
import itertools
import time
import zipfile
import math
import sys
import collections
from datetime import timedelta, datetime, date
from decimal import Decimal
import numpy as np

try:
    from cdecimal import Decimal as cDecimal
except:
    cDecimal = Decimal

from chorde.clients.inproc import Cache

import cython

npuint64 = cython.declare(object, numpy.uint64)
npint64 = cython.declare(object, numpy.int64)
npuint32 = cython.declare(object, numpy.uint32)
npint32 = cython.declare(object, numpy.int32)
npfloat64 = cython.declare(object, numpy.float64)
npfloat32 = cython.declare(object, numpy.float32)

if cython.compiled:
    # Compatibility fix for cython >= 0.23, which no longer supports "buffer" as a built-in type
    buffer = cython.declare(object, buffer)  # lint:ok
    from types import BufferType as buffer

    assert Py_LT == 0
    assert Py_LE == 1
    assert Py_EQ == 2
    assert Py_NE == 3
    assert Py_GT == 4
    assert Py_GE == 5

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
    """
    Takes a buffer object as parameter and returns a writable object with buffer protocol.
    """
    if type(buf) is buffer or type(buf) is bytearray or type(buf) is bytes or isinstance(buf, bytes):
        return buf
    else:
        return buffer(buf)

@cython.inline
@cython.cfunc
def _likerobuffer(buf):
    """
    Takes a buffer object as parameter and returns a read-only object with buffer protocol.
    """
    if type(buf) is buffer or type(buf) is bytes or isinstance(buf, bytes):
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
    @cython.locals(
        i=int, j=int, offs=cython.longlong,
        pybuf='Py_buffer', pbuf='const unsigned char *', b=cython.uchar)
    def unpack_from(cls, buf, offs, idmap = None):
        buf = _likerobuffer(buf)
        if cython.compiled:
            PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)  # lint:ok
            pbuf = cython.cast(cython.p_uchar, pybuf.buf)  # lint:ok
            if offs >= pybuf.len:
                PyBuffer_Release(cython.address(pybuf))  # lint
                raise IndexError("Offset out of range")
        else:
            pbuf = buf
        try:
            if pbuf[offs] == 'm':
                # inline bitmap
                if cython.compiled and offs+7 >= pybuf.len:
                    raise IndexError("Object spans beyond buffer end")
                rv = []
                for i in xrange(7):
                    b = ord(pbuf[offs+1+i])
                    if b:
                        for j in xrange(8):
                            if b & (1<<j):
                                rv.append(i*8+j)
                return frozenset(rv)
            else:
                # unpack a list, build a set from it
                return frozenset(mapped_list.unpack_from(buf, offs, idmap))
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
                # inline unsigned ints
                buf[offs] = dtype = 'I'
            elif -0x80000000 <= maxval <= 0x7FFFFFFF:
                # inline signed ints
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
                        idmap[xid] = val_offs = offs + implicit_offs
                        mx = mapped_object(x)
                        offs = mx.pack_into(mx, buf, offs, idmap, implicit_offs)
                    else:
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
    @cython.locals(rv = list)
    def unpack_from(cls, buf, offs, idmap = None, array = array.array, itemsizes = {
                dtype : array.array(dtype, []).itemsize
                for dtype in ('B','b','H','h','I','i','l','d')
            } ):
        if idmap is None:
            idmap = {}
        if offs in idmap:
            return idmap[offs]

        baseoffs = offs
        buf = _likerobuffer(buf)
        dcode = buf[offs]
        if dcode in ('B','b','H','h','I','i'):
            dtype = dcode
            objlen, = struct.unpack('<I', buf[offs+1:offs+4] + '\x00')
            offs += 4
            if objlen == 0xFFFFFF:
                objlen = struct.unpack_from('<Q', buf, offs)
                offs += 8
            rv = list(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 'q':
            dtype = 'l'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8
            rv = list(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 'd':
            dtype = 'd'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8
            rv = list(array(dtype, buf[offs:offs+itemsizes[dtype]*objlen]))
        elif dcode == 't':
            dtype = 'l'
            objlen, = struct.unpack('<Q', buf[offs+1:offs+8] + '\x00')
            offs += 8

            index = array(dtype, buf[offs:offs+itemsizes[dtype]*objlen])

            idmap[baseoffs] = rv = ([None] * objlen)
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

class mapped_dict(dict):

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        return proxied_dict.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        proxy = proxied_dict.unpack_from(buf, offs, idmap)
        return proxy.copy()


@cython.cfunc
@cython.locals(code=cython.ulonglong, nbits=cython.ulonglong)
@cython.returns(cython.ulonglong)
def _hash_rotl(code, nbits):
    return (code << nbits) | (code >> (64 - nbits))


@cython.cfunc
@cython.locals(code1=cython.ulonglong, code2=cython.ulonglong)
@cython.returns(cython.ulonglong)
def _mix_hash(code1, code2):
    return _hash_rotl(code1, 5) ^ code2


_TUPLE_SEED = 1626619511096549620
_FSET_SEED  = 8212431769940327799

@cython.locals(hval=cython.ulonglong)
def _stable_hash(key):
    if isinstance(key, basestring):
        hval = xxhash.xxh64(safe_utf8(key)).intdigest()
    elif isinstance(key, (int, long)):
        hval = key
    elif isinstance(key, float):
        trunc_key = int(key)
        if trunc_key == key:
            hval = trunc_key
        else:
            mant, expo = math.frexp(key)
            hval = _mix_hash(expo, int(mant * 0xffffffffffff))
    elif isinstance(key, (tuple, frozenset, proxied_tuple)):
        if isinstance(key, frozenset):
            hval = _FSET_SEED
        else:
            hval = _TUPLE_SEED

        for value in key:
            hval = _mix_hash(hval, _stable_hash(value))
    else:
        raise TypeError("unhashable type: %s" % type(key).__name__)

    return hval if hval != 0 else 1


@cython.locals(idx=int)
def _enum_keys(obj):
    for idx, key in enumerate(obj.iterkeys()):
        yield key, idx


class BufferIO(object):
    """
    Gimmick object. Its whole purpose is to implement
    the protocol expected by the IdMapper classes.
    """

    def __init__(self, buf, offs):
        self.buf = buf
        self.offs = offs
        self.pos = 0

    def write(self, value):
        size = len(value)
        offs = self.offs + self.pos
        self.buf[offs:offs + size] = value
        self.pos += size
        return size

    def tell(self):
        return self.pos

    def flush(self):
        pass

    def seek(self, pos):
        self.pos = pos


@cython.cclass
class proxied_dict(object):

    HEADER_PACKER = struct.Struct('=Q')   # Offset into value list.

    cython.declare(objmapper=object, vlist=proxied_list)

    def __init__(self, objmapper, vlist):
        self.objmapper = objmapper
        self.vlist = vlist

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        packer = cls.HEADER_PACKER
        ipos = offs
        offs += packer.size
        iobuf = BufferIO(buf, offs)
        offs += ObjectIdMapper.build(_enum_keys(obj), iobuf, return_mapper=False)
        packer.pack_into(buf, ipos, iobuf.tell())
        return offs + proxied_list.pack_into(obj.values(), buf, offs, idmap, implicit_offs)

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        packer = cls.HEADER_PACKER
        mapper_size, = packer.unpack_from(buf, offs)
        offs += packer.size
        objmapper = ObjectIdMapper.map_buffer(buf, offs)
        offs += mapper_size
        vlist = proxied_list.unpack_from(buf, offs, idmap)
        return proxied_dict(objmapper, vlist)

    def get(self, key, default_val = None):
        idx = self.objmapper.get(key)
        return self.vlist[idx] if idx is not None else default_val

    def __getitem__(self, key):
        idx = self.objmapper.get(key)
        if idx is None:
            raise KeyError(key)
        return self.vlist[idx]

    def __contains__(self, key):
        return key in self.objmapper

    def has_key(self, key):
        return key in self

    def iteritems(self):
        for key, idx in self.objmapper.iteritems():
            yield key, self.vlist[idx]

    def items(self):
        return list(self.iteritems())

    def iterkeys(self):
        return iter(self.objmapper)

    def keys(self):
        return self.objmapper.keys()

    def itervalues(self):
        return iter(self.vlist)

    def values(self):
        return self.vlist

    def __iter__(self):
        return self.iterkeys()

    def __len__(self):
        return len(self.vlist)

    def copy(self):
        return dict(self.iteritems())

    def viewkeys(self):
        return self.keys()

    def viewvalues(self):
        return self.values()

    def viewitems(self):
        return self.items()

    def __setitem__(self, index, value):
        raise TypeError("Proxy objects are read-only")

    def __delitem__(self, index):
        raise TypeError("Proxy objects are read-only")

    def _is_eq(self, other):
        if len(self) != len(other):
            return False

        for key, val in self.iteritems():
            if key not in other or other[key] != val:
                return False

        return True

    def __richcmp__(self, other, op):
        if op != 2 and op != 3:   # != Py_EQ && != Py_NE
            diff = id(self) - id(other)
            if op == 0:     # Py_LT
                return diff < 0
            elif op == 1:   # Py_LE
                return diff <= 0
            elif op == 4:   # Py_GT
                return diff > 0
            elif op == 5:   # Py_GE
                return diff >= 0

            return False   # Shouldn't happen

        rv = self._is_eq(other)
        return rv if op == 2 else not rv

    if not cython.compiled:
        def _cmp(self, other):
            return cmp(id(self), id(other))

        def _eq(self, other):
            return self._is_eq(other)

        def _ne(self, other):
            return not self._is_eq(other)

    def __repr__(self):
        return "proxied_dict(%s)" % self

    def __str__(self):
        return "{%s}" % ", ".join(["%r: %r" % (k, v) for k, v in self.iteritems()])


if not cython.compiled:
    for orig, new in (('_cmp', '__cmp__'), ('_eq', '__eq__'), ('_ne', '__ne__')):
        setattr(proxied_dict, new, getattr(proxied_dict, orig))
    delattr(proxied_dict, '__richcmp__')

class proxied_buffer(object):

    HEADER_PACKER = struct.Struct('=Q')

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        packer = cls.HEADER_PACKER
        packer.pack_into(buf, offs, len(obj))
        offs += packer.size

        end_offs = offs + len(obj)
        buf[offs:end_offs] = obj

        return end_offs

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        packer = cls.HEADER_PACKER
        size, = packer.unpack_from(buf, offs)
        offs += packer.size

        return buffer(buf, offs, size)

class proxied_ndarray(object):

    HEADER_PACKER = struct.Struct('=QQ')

    @classmethod
    def _make_dtype_params(cls, dtype):
        names = dtype.names
        if names:
            # It is a Structured array
            fields = dtype.fields
            return [
                (k, cls._make_dtype_params(fields[k][0]))
                for k in names
            ]
        else:
            return dtype.str

    @classmethod
    @cython.locals(offs = cython.ulonglong, implicit_offs = cython.ulonglong, header_offs = cython.ulonglong)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        header_offs = offs
        packer = cls.HEADER_PACKER
        offs += packer.size

        offs = mapped_tuple.pack_into(obj.shape, buf, offs)
        dtype_offs = offs - header_offs

        dtype_params = cls._make_dtype_params(obj.dtype)
        if isinstance(dtype_params, str):
            dtype_params = [dtype_params]

        offs = mapped_list.pack_into(dtype_params, buf, offs)
        data_offs = offs - header_offs

        packer.pack_into(buf, header_offs, dtype_offs, data_offs)
        return proxied_buffer.pack_into(buffer(obj), buf, offs)


    @classmethod
    @cython.locals(offs = cython.ulonglong, dtype_offs = cython.ulonglong, data_offs = cython.ulonglong)
    def unpack_from(cls, buf, offs, idmap = None):
        packer = cls.HEADER_PACKER
        dtype_offs, data_offs = packer.unpack_from(buf, offs)

        shape = mapped_tuple.unpack_from(buf, offs + packer.size)
        dtype_params = mapped_list.unpack_from(buf, offs + dtype_offs)
        if isinstance(dtype_params[0], str):
            dtype_params = dtype_params[0]

        data = proxied_buffer.unpack_from(buf, offs + data_offs)

        ndarray = np.frombuffer(data, np.dtype(dtype_params))
        return ndarray.reshape(shape)

# @cython.ccall
# @cython.returns(cython.bint)
@cython.locals(_cmp = int)
def proxied_list_richcmp(a, b, op):

    if not islist(a) or not islist(b):
        if op == 2: # Py_EQ
            return False
        elif op == 3: # Py_NE
            return True
        else:
            raise NotImplementedError

    _cmp = proxied_list_cmp(a, b)
    if op == 2: # Py_EQ
        return _cmp == 0
    elif op == 3: # Py_NE
        return _cmp != 0
    elif op == 0: # Py_LT
        return _cmp < 0
    elif op == 1: # Py_LE
        return _cmp <= 0
    elif op == 4: # Py_GT
        return _cmp > 0
    elif op == 5: # op == Py_GE:
        return _cmp >= 0
    else:
        raise NotImplementedError

#@cython.ccall
#@cython.returns(int)
def proxied_list_cmp(a, b):

    alen = len(a)
    blen = len(b)

    for i in xrange(min(alen, blen)):
        selfe = a[i]
        othere = b[i]

        if selfe < othere:
            return -1
        elif selfe > othere:
            return 1

    if alen < blen:
        return -1
    elif alen > blen:
        return 1

    return 0

@cython.ccall
@cython.returns(cython.bint)
def islist(obj):
    return isinstance(obj, (tuple, list, proxied_list, proxied_tuple))

@cython.cclass
class proxied_list(object):

    cython.declare(
        buf = object,
        pybuf = 'Py_buffer',
        offs = cython.ulonglong,
        elem_start = cython.longlong,
        elem_end = cython.longlong,
        elem_step = cython.longlong
    )

    if cython.compiled:
        def __del__(self):
            if self.pybuf.buf != cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok

    @cython.ccall
    @cython.locals(dataoffs = cython.ulonglong, dcode = cython.char, pbuf = 'const char *',
        itemsize = cython.uchar, objlen = cython.ulonglong)
    def _metadata(self,
        itemsizes = dict([(dtype, array.array(dtype, []).itemsize) for dtype in ('B','b','H','h','I','i','l','d')])):

        if cython.compiled:
            # Cython version
            dataoffs = self.offs
            pbuf = cython.cast(cython.p_char, self.pybuf.buf)
            dcode = pbuf[dataoffs]

            if dcode in ('B','b','H','h','I','i'):

                objlen = cython.cast(cython.p_uint, pbuf + dataoffs)[0] >> 8
                dataoffs += 4

                if objlen == 0xFFFFFF:
                    objlen = cython.cast(cython.p_longlong, pbuf + dataoffs)[0]
                    dataoffs += 8

                if dcode in ('B', 'b'):
                    itemsize = 1
                elif dcode in ('H', 'h'):
                    itemsize = 2
                else:
                    itemsize = 4

                return dcode, objlen, itemsize, dataoffs, None

            elif dcode in ('q', 'd', 't'):
                objlen = cython.cast(cython.p_longlong, pbuf + dataoffs)[0] >> 8
                dataoffs += 8
                return dcode, objlen, 8, dataoffs, None

            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))

        else:
            # Python version
            dataoffs = self.offs
            buf = self.buf
            dcode = buf[dataoffs]

            if dcode in ('B','b','H','h','I','i'):
                objlen, = struct.unpack('<I', buf[dataoffs+1:dataoffs+4] + '\x00')
                dataoffs += 4
                if objlen == 0xFFFFFF:
                    objlen = struct.unpack_from('<Q', buf, dataoffs)
                    dataoffs += 8
                return dcode, objlen, itemsizes[dcode], dataoffs, struct.Struct(dcode)

            elif dcode == 'q':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['l'], dataoffs, struct.Struct('l')

            elif dcode == 'd':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['d'], dataoffs, struct.Struct('d')

            elif dcode == 't':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['l'], dataoffs, struct.Struct('l')

            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))

    @cython.locals(offs = cython.ulonglong, idmap = dict, elem_start = cython.longlong,
        elem_end = cython.longlong, elem_step = cython.longlong)
    def __init__(self, buf, offs, idmap = None, elem_start = 0, elem_end = 0, elem_step = 0):
        self.offs = offs
        self.buf = buf

        if elem_step < 0:
            elem_end = min(elem_end, elem_start)
        elif elem_step > 0:
            elem_start = min(elem_start, elem_end)

        self.elem_start = elem_start
        self.elem_end = elem_end
        self.elem_step = elem_step
        self._init()

    @cython.ccall
    def _init(self):
        if cython.compiled:
            self.pybuf.buf = cython.NULL
            PyObject_GetBuffer(self.buf, cython.address(self.pybuf), PyBUF_SIMPLE)  # lint:ok

        # Call metadata to check the object
        self._metadata()

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        # Same format as tuple, only different base type
        return mapped_tuple.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):

        if idmap is None:
            idmap = {}
        if offs in idmap:
            return idmap[offs]

        buf = _likerobuffer(buf)
        return cls(buf, offs, idmap)

    @cython.locals(obj_offs = cython.ulonglong, dcode = cython.char, index = cython.longlong, pindex = "unsigned long *",
        dataoffs = cython.ulonglong, objlen = cython.longlong, xlen = cython.longlong, step = cython.longlong)
    def _getitem(self, index):

        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        xlen = objlen
        step = 1

        if self.elem_step != 0:
            if self.elem_end == self.elem_start:
                raise IndexError(index)
            step = abs(self.elem_step)
            if self.elem_step > 0:
                xlen = (self.elem_end - self.elem_start - 1) / step + 1
            else:
                xlen = (self.elem_start - self.elem_end - 1) / step + 1

            index = self.elem_start + index * self.elem_step

            if (self.elem_step < 0 and (index > self.elem_start or index <= self.elem_end)) or (
                self.elem_step > 0 and (index >= self.elem_end or index < self.elem_start)):
                raise IndexError(index)

        if index < 0:
            index += xlen

        if index >= objlen or index < 0:
            raise IndexError(index)

        if dcode == 't':
            if cython.compiled:
                pindex = cython.cast(cython.p_ulong, cython.cast(cython.p_uchar, self.pybuf.buf) + dataoffs)
                obj_offs = self.offs + pindex[index]
            else:
                index_offs = dataoffs + itemsize * int(index)
                obj_offs = self.offs + _struct.unpack(self.buf[index_offs:index_offs + itemsize])[0]
        else:
            obj_offs = dataoffs + itemsize * int(index)

        if dcode == 't':
            res = mapped_object.unpack_from(self.buf, obj_offs)
        elif cython.compiled:
            if dcode == 'B':
                res = cython.cast(cython.p_uchar,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'b':
                res = cython.cast(cython.p_char,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'H':
                res = cython.cast(cython.p_ushort,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'h':
                res = cython.cast(cython.p_short,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'I':
                res = cython.cast(cython.p_uint,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'i':
                res = cython.cast(cython.p_int,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'q':
                res = cython.cast(cython.p_longlong,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'd':
                res = cython.cast(cython.p_double,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
        else:
            res = _struct.unpack(self.buf[obj_offs:obj_offs + itemsize])[0]

        return res

    @cython.ccall
    def _make_empty(self):
        return []

    def __getitem__(self, index):
        if isinstance(index, slice):
            xlen = len(self)
            start, end, step = index.indices(xlen)
            if self.elem_step != 0:
                start = self.elem_start + start * self.elem_step
                end = self.elem_start + end * self.elem_step
                step *= self.elem_step

            if (step < 0 and end >= start) or (step >= 0 and start >= end):
                return self._make_empty()

            return proxied_tuple(
                buf = self.buf,
                offs = self.offs,
                idmap = None,
                elem_start = start,
                elem_end = end,
                elem_step = step)

        return self._getitem(index)

    @cython.locals(op = cython.char)
    def  __richcmp__(self, other, op):
        return proxied_list_richcmp(self, other, op)

    if not cython.compiled:
        def __cmp__(self, other):
            if not islist(other):
                raise NotImplementedError
            return proxied_list_cmp(self, other)

        def _ne(self, other):
            if not islist(other):
                return True
            return proxied_list_cmp(self, other) != 0

        def _eq(self, other):
            if not islist(other):
                return False
            return proxied_list_cmp(self, other) == 0

    def __len__(self):
        if self.elem_step == 0:
            return self._metadata()[1]
        elif self.elem_start == self.elem_end:
            return 0
        elif self.elem_step < 0:
            return (self.elem_start - self.elem_end - 1) / (-self.elem_step) + 1
        else:
            return (self.elem_end - self.elem_start - 1) / self.elem_step + 1

    def __nonzero__(self):
        return len(self) > 0

    def __bool__(self):
        return len(self) > 0

    def __setitem__(self, index, value):
        raise TypeError("Proxy objects are read-only")

    def __delitem__(self, index):
        raise TypeError("Proxy objects are read-only")

    @cython.locals(i=cython.longlong)
    def __iter__(self):
        for i in xrange(len(self)):
            yield self[i]

    @cython.locals(l=cython.longlong)
    def __reversed__(self):
        l = len(self)
        if l > 0:
            for i in xrange(l - 1, -1, -1):
                yield self[i]

    def __contains__(self, item):
        for e in self:
            if e == item:
                return True
        return False

    def __repr__(self):
        return "proxied_list(%s)" % self

    def __str__(self):
        return "[%s]" % ",".join([str(x) for x in self])

if not cython.compiled:
    setattr(proxied_list, '__eq__', getattr(proxied_list, '_eq'))
    setattr(proxied_list, '__ne__', getattr(proxied_list, '_ne'))

is_cpython = cython.declare(cython.bint, sys.subversion[0] == 'CPython')

@cython.cclass
class proxied_tuple(proxied_list):

    cython.declare(
        _hash = cython.long,
    )

    def __init__(self, *args, **kwargs):
        super(proxied_tuple, self).__init__(*args, **kwargs)
        self._hash = -1

    @cython.ccall
    def _make_empty(self):
        return ()

    @cython.locals(mult = cython.long, x = cython.long, y = cython.long, len_ = cython.Py_ssize_t, i = cython.Py_ssize_t)
    def __hash__(self):
        if self._hash == -1:
            if cython.compiled and is_cpython:
                # From Python 2.7 source code
                mult = 1000003
                x = 0x345678
                len_ = len(self)
                for i in xrange(len_):
                    len_ -= 1
                    y = hash(self[i])
                    x = (x ^ y) * mult;
                    mult += 82520 + len_ + len_;

                x += 97531
                if x == -1:
                    x = -2
                self._hash = x
            else:
                self._hash = hash(tuple(iter(self)))
        return self._hash

    @cython.locals(op = cython.char)
    def  __richcmp__(self, other, op):
        return proxied_list_richcmp(self, other, op)

    def __repr__(self):
        return "proxied_tuple(%s)" % self

    def __str__(self):
        return "(%s)" % ",".join([str(x) for x in self])
try:
    import lz4.block as lz4_block
except ImportError:
    import lz4 as lz4_block

lz4_decompress = cython.declare(object, lz4_block.decompress)
lz4_compress = cython.declare(object, lz4_block.compress)

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
        offs = int(offs)
        dataoffs = offs + hpacker.size
        compressed = (objlen & 0x8000) != 0
        if (objlen & 0x7FFF) == 0x7FFF:
            qpacker = struct.Struct('=HQ')
            objlen = qpacker.unpack_from(buf, offs)[1]
            dataoffs = offs + qpacker.size
        else:
            objlen = objlen & 0x7FFF
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
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        rv = mapped_bytes.unpack_from(buf, offs).decode("utf8")
        if idmap is not None:
            idmap[offs] = rv
        return rv

class mapped_decimal(Decimal):
    PACKER = struct.Struct('=q')

    @classmethod
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, exponent = cython.longlong, sign = cython.uchar)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = id(obj)
            idmap[objid] = offs + implicit_offs

        if not isinstance(obj, (Decimal, cDecimal)):
            obj = cDecimal(obj)

        packer = cls.PACKER
        sign, digits, exponent = obj.as_tuple()
        packer.pack_into(buf, offs, (exponent << 1) | sign)
        offs += packer.size

        return mapped_tuple.pack_into(digits, buf, offs, None, implicit_offs)

    @classmethod
    @cython.locals(offs = cython.longlong, exponent = cython.longlong, sign = cython.uchar)
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        packer = cls.PACKER
        exponent, = packer.unpack_from(buf, offs)
        sign = exponent & 0x1

        digits = mapped_tuple.unpack_from(buf, offs + packer.size, idmap)
        rv = cDecimal((sign, digits, exponent >> 1))

        if idmap is not None:
            idmap[offs] = rv
        return rv

class mapped_datetime(datetime):
    PACKER = struct.Struct('=q')

    @classmethod
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, timestamp = cython.longlong)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = id(obj)
            idmap[objid] = offs + implicit_offs

        packer = cls.PACKER
        timestamp = int(time.mktime(obj.timetuple()))
        packer.pack_into(buf, offs, (timestamp << 20) + obj.microsecond)

        return offs + packer.size

    @classmethod
    @cython.locals(offs = cython.longlong, timestamp = cython.longlong, microseconds = cython.ulong)
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        packer = cls.PACKER
        timestamp, = packer.unpack_from(buf, offs)
        microseconds = timestamp & 0xFFFFF
        rv =  datetime.fromtimestamp(timestamp >> 20) + timedelta(microseconds=microseconds)

        if idmap is not None:
            idmap[offs] = rv
        return rv

class mapped_date(date):
    PACKER = struct.Struct('=q')

    @classmethod
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, timestamp = cython.longlong)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = id(obj)
            idmap[objid] = offs + implicit_offs

        packer = cls.PACKER
        timestamp = int(time.mktime(obj.timetuple()))
        packer.pack_into(buf, offs, timestamp)

        return offs + packer.size

    @classmethod
    @cython.locals(offs = cython.longlong, timestamp = cython.longlong)
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        packer = cls.PACKER
        timestamp, = packer.unpack_from(buf, offs)
        rv =  date.fromtimestamp(timestamp)

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
        mapped_unicode : 'u',
        mapped_dict : 'm',
        mapped_bytes : 's',
        mapped_datetime : 'v',
        mapped_date : 'V',
        mapped_decimal : 'F',

        proxied_list: 'E',
        proxied_tuple: 'W',
        proxied_ndarray: 'n',
        proxied_buffer: 'r',
        proxied_dict: 'M',

        int : 'q',
        long : 'q',
        float : 'd',
        str : 's',
        unicode : 'u',
        datetime : 'v',
        date : 'V',
        Decimal : 'F',
        cDecimal : 'F',
        np.ndarray : 'n',
        buffer : 'r',

        dict : 'm',
        collections.defaultdict : 'm',
        set : 'Z',
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
        'm' : (mapped_dict.pack_into, mapped_dict.unpack_from, mapped_dict),
        'n' : (proxied_ndarray.pack_into, proxied_ndarray.unpack_from, proxied_ndarray),
        'r' : (proxied_buffer.pack_into, proxied_buffer.unpack_from, proxied_buffer),
        'W' : (proxied_tuple.pack_into, proxied_tuple.unpack_from, proxied_tuple),
        'E' : (proxied_list.pack_into, proxied_list.unpack_from, proxied_list),
        'v' : (mapped_datetime.pack_into, mapped_datetime.unpack_from, mapped_datetime),
        'V' : (mapped_date.pack_into, mapped_date.unpack_from, mapped_date),
        'F' : (mapped_decimal.pack_into, mapped_decimal.unpack_from, mapped_decimal),
        'M' : (proxied_dict.pack_into, proxied_dict.unpack_from, proxied_dict)
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
        if typecode is not None and typ in cls.TYPE_CODES:
            if cls.TYPE_CODES[typ] != typecode or cls.OBJ_PACKERS[typecode][2].schema is not schema:
                raise ValueError("Registering different types with same typecode %r: %r" % (cls.TYPE_CODES[typ], typ))
            return cls.OBJ_PACKERS[typecode][2]

        if isinstance(typ, mapped_object_with_schema):
            packable = typ
        else:
            packable = mapped_object_with_schema(schema)
        class SchemaBufferProxyProperty(GenericBufferProxyProperty):
            typ = packable
        if typecode is not None:
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
mapped_object.OBJ_PACKERS['o'] = (mapped_object.pack_into, mapped_object.unpack_from, mapped_object)

VARIABLE_TYPES = {
    frozenset : mapped_frozenset,
    set : mapped_frozenset,
    tuple : mapped_tuple,
    list : mapped_list,
    dict : mapped_dict,
    collections.defaultdict : mapped_dict,
    str : mapped_bytes,
    unicode : mapped_unicode,
    bytes : mapped_bytes,
    object : mapped_object,
    datetime : mapped_datetime,
    date : mapped_date,
    Decimal : mapped_decimal,
    cDecimal : mapped_decimal,
    np.ndarray : proxied_ndarray,
    buffer : proxied_buffer,
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
class BufferProxyObject(object):
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
class BaseBufferProxyProperty(object):
    cython.declare(offs = cython.ulonglong, mask = cython.ulonglong)

    def __init__(self, offs, mask):
        self.offs = offs
        self.mask = mask

    def __set__(self, obj, value):
        raise TypeError("Proxy objects are read-only")

    def __delete__(self, obj):
        raise TypeError("Proxy objects are read-only")

@cython.cclass
class BoolBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar) if cython.compiled else struct.Struct('B').size

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
            return struct.unpack_from('B', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class UByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar) if cython.compiled else struct.Struct('B').size

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
            return struct.unpack_from('B', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class ByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.char) if cython.compiled else struct.Struct('b').size

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
            return struct.unpack_from('b', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class UShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.ushort) if cython.compiled else struct.Struct('H').size

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
            return struct.unpack_from('H', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class ShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.short) if cython.compiled else struct.Struct('h').size

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
            return struct.unpack_from('h', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class UIntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uint) if cython.compiled else struct.Struct('I').size

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
            return struct.unpack_from('I', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class IntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.int) if cython.compiled else struct.Struct('i').size

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
            return struct.unpack_from('i', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class ULongBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.ulonglong) if cython.compiled else struct.Struct('Q').size

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
            return struct.unpack_from('Q', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class LongBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

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
            return struct.unpack_from('q', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class DoubleBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.double) if cython.compiled else struct.Struct('d').size

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
            return struct.unpack_from('d', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class FloatBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.float) if cython.compiled else struct.Struct('f').size

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
            return struct.unpack_from('f', obj.buf, obj.offs + self.offs)[0]

@cython.cclass
class BytesBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

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
            offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)[0]
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
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

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
            offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)[0]
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
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

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
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

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
            poffs = offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)[0]
        rv = self.typ.unpack_from(obj.buf, offs)
        if obj.idmap is not None:
            obj.idmap[poffs] = rv
        return rv

@cython.cclass
class FrozensetBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_frozenset

@cython.cclass
class DictBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_dict

@cython.cclass
class ProxiedNDArrayBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_ndarray

@cython.cclass
class ProxiedBufferBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_buffer

@cython.cclass
class DatetimeBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_datetime

@cython.cclass
class DateBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_date

@cython.cclass
class DecimalBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_decimal

@cython.cclass
class TupleBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_tuple

@cython.cclass
class ListBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_list

@cython.cclass
class ProxiedTupleBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_tuple

@cython.cclass
class ProxiedListBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_list

@cython.cclass
class ObjectBufferProxyProperty(GenericBufferProxyProperty):
    typ = mapped_object

@cython.cclass
class ProxiedDictBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_dict

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
    mapped_dict : DictBufferProxyProperty,
    mapped_bytes : BytesBufferProxyProperty,
    mapped_unicode : UnicodeBufferProxyProperty,
    mapped_bytes : BytesBufferProxyProperty,
    mapped_object : ObjectBufferProxyProperty,
    mapped_datetime : DatetimeBufferProxyProperty,
    mapped_date : DateBufferProxyProperty,
    mapped_decimal : DecimalBufferProxyProperty,

    proxied_tuple : ProxiedTupleBufferProxyProperty,
    proxied_list : ProxiedListBufferProxyProperty,
    proxied_ndarray : ProxiedNDArrayBufferProxyProperty,
    proxied_buffer : ProxiedBufferBufferProxyProperty,
    proxied_dict : ProxiedDictBufferProxyProperty,

    int : LongBufferProxyProperty,
    long : LongBufferProxyProperty,
    float : DoubleBufferProxyProperty,
    str : BytesBufferProxyProperty,
    unicode : UnicodeBufferProxyProperty,
    datetime : DatetimeBufferProxyProperty,
    date : DateBufferProxyProperty,
    Decimal : DecimalBufferProxyProperty,
    cDecimal : DecimalBufferProxyProperty,
    np.ndarray : ProxiedNDArrayBufferProxyProperty,
    buffer : ProxiedBufferBufferProxyProperty,
}

def GenericProxyClass(slot_keys, slot_types, present_bitmap, base_offs, bases = None):
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
    if bases is not None:
        GenericProxyClass.__bases__ += bases

    return GenericProxyClass

@cython.cclass
class Schema(object):
    cython.declare(
        slot_types = dict,
        pack_buffer_size = int,
        max_pack_buffer_size = int,
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
        _proxy_bases = tuple,

        _pack_buffer = object,
        _var_bitmap = cython.ulonglong,
        _fixed_bitmap = cython.ulonglong,
        _last_unpacker = tuple,
        _last_unpacker_bitmap = cython.ulonglong,
    )

    @property
    def Proxy(self):
        return functools.partial(self._Proxy, "\x00" * self.bitmap_size, 0, 0, None)

    @property
    def ProxyClass(self):
        return self._Proxy

    def __init__(self, slot_types, alignment = 8, pack_buffer_size = 65536, packer_cache = None, unpacker_cache = None,
            max_pack_buffer_size = None):
        self.init(
            self._map_types(slot_types),
            packer_cache = packer_cache, unpacker_cache = unpacker_cache, alignment = alignment,
            pack_buffer_size = pack_buffer_size)

    def __reduce__(self):
        return (type(self), (self.slot_types,), self.__getstate__())

    def __getstate__(self):
        return dict(
            slot_types = self.slot_types,
            slot_keys = self.slot_keys,
            alignment = self.alignment,
            bases = self._proxy_bases,
        )

    def __setstate__(self, state):
        # These are a safety hazard (they could cause OOM if corrupted), so ignore them if present
        # Not needed for unpacking anyway
        state.pop('pack_buffer_size', None)
        state.pop('max_pack_buffer_size', None)
        bases = state.pop('bases', None)
        state['autoregister'] = True

        self.init(**state)

        if bases is not None:
            self.set_proxy_bases(bases)

    def set_proxy_bases(self, bases):
        self._proxy_bases = bases

    @cython.locals(other_schema = 'Schema')
    def compatible(self, other):
        if not isinstance(other, Schema):
            return False

        other_schema = other
        if self.slot_keys != other_schema.slot_keys or self.alignment != other_schema.alignment:
            return False

        for k in self.slot_keys:
            if self.slot_types[k] is not other_schema.slot_types[k]:
                return False

        return True

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

    @cython.locals(slot_types = dict, slot_keys = tuple)
    def init(self, slot_types = None, slot_keys = None, alignment = 8, pack_buffer_size = 65536,
            max_pack_buffer_size = None, packer_cache = None, unpacker_cache = None,
            autoregister = False):
        # Freeze slot order, sort by descending size to optimize alignment
        self._proxy_bases = None
        if slot_types is None:
            slot_types = self.slot_types

        # Map compatible types
        for k, typ in slot_types.iteritems():
            if not isinstance(typ, mapped_object_with_schema):
                continue
            if typ in mapped_object.TYPE_CODES:
                continue

            typ_schema = getattr(typ, 'schema', None)
            if typ_schema is None:
                continue

            # Find compatible type
            for packer, unpacker, packable_type in mapped_object.OBJ_PACKERS.itervalues():
                if not isinstance(packable_type, mapped_object_with_schema):
                    continue

                packable_schema = getattr(packable_type, 'schema', None)
                if packable_schema is None:
                    continue

                if packable_schema.compatible(typ_schema):
                    slot_types[k] = packable_type
                    break
            else:
                # No compatible schema found, register if we're in autoregister mode
                # without a typecode (ie: only for explicitly typed references)
                if autoregister:
                    mapped_object.register_schema(typ, typ_schema, None)

        if slot_keys is None:
            slot_keys = tuple(
                sorted(
                    slot_types.keys(),
                    key = lambda k, sget = slot_types.get, fget = FIXED_TYPES.get :
                        (-struct.Struct(fget(sget(k), 'q')).size, k)
                )
            )

        self.alignment = alignment
        self.pack_buffer_size = pack_buffer_size

        if packer_cache is None:
            packer_cache = Cache(256)
        if unpacker_cache is None:
            unpacker_cache = Cache(256)
        if max_pack_buffer_size is not None:
            self.max_pack_buffer_size = max_pack_buffer_size
        else:
            self.max_pack_buffer_size = max(128<<20, max(pack_buffer_size, min(pack_buffer_size * 2, 0x7FFFFFFF)))

        self.packer_cache = packer_cache
        self.unpacker_cache = unpacker_cache
        self.slot_types = slot_types
        self.slot_keys = slot_keys
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
                fixed_bitmap |= cython.cast(cython.ulonglong, 1) << i
            else:
                var_bitmap |= cython.cast(cython.ulonglong, 1) << i
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
        self._Proxy = GenericProxyClass(self.slot_keys, self.slot_types, 0, self.bitmap_size, self._proxy_bases)

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int)
    @cython.returns(tuple)
    def _get_bitmaps(self, obj):
        has_bitmap = 0
        none_bitmap = 0
        for i,slot in enumerate(self.slot_keys):
            if hasattr(obj, slot):
                has_bitmap |= cython.cast(cython.ulonglong, 1) << i
                if getattr(obj, slot, 0) is None:
                    none_bitmap |= cython.cast(cython.ulonglong, 1) << i
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
                if present_bitmap & (cython.cast(cython.ulonglong, 1) << i)
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
                if present_bitmap & (cython.cast(cython.ulonglong, 1) << i)
            ])
            unpacker = struct.Struct(pformat)
            alignment = self.alignment
            size = unpacker.size
            padding = (size + self.bitmap_size + alignment - 1) / alignment * alignment - size
            gfactory = GenericProxyClass(
                self.slot_keys, self.slot_types, present_bitmap, self.bitmap_size,
                self._proxy_bases)
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
            mask = cython.cast(cython.ulonglong, 1) << i
            if present_bitmap & mask:
                val = getattr(obj, slot)
                if fixed_present & mask:
                    packable_append(val)
                else:
                    val_id = id(val)
                    val_offs = idmap_get(val_id)
                    if val_offs is None:
                        idmap[val_id] = ival_offs = offs + implicit_offs
                        try:
                            offs = slot_types[slot].pack_into(val, buf, offs, idmap, implicit_offs)
                        except Exception as e:
                            try:
                                # Add some context. It may not work with all exception types, hence the fallback
                                e = type(e)("%s packing attribute %s=%r of type %r" % (
                                    e, slot, val, type(obj).__name__))
                            except:
                                pass
                            else:
                                raise e
                            raise
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
        try:
            packer.pack_into(buf, baseoffs, *packable)
        except struct.error as e:
            raise struct.error("%s packing %r with format %r for type %s" % (
                e,
                packable,
                packer.format,
                type(obj).__name__,
            ))
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
                if len(self._pack_buffer) >= self.max_pack_buffer_size:
                    raise
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
        else:
            offs = int(offs)

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
                        mask = cython.cast(cython.ulonglong, 1) << i
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

class mapped_object_with_schema(object):
    cython.declare(schema = Schema)

    def __init__(self, schema):
        self.schema = schema

    def pack_into(self, obj, buf, offs, idmap = None, implicit_offs = 0):
        return self.schema.pack_into(obj, buf, offs, idmap, implicit_offs = implicit_offs)

    def unpack_from(self, buf, offs, idmap = None):
        return self.schema.unpack_from(buf, offs, idmap)

@cython.ccall
def _map_zipfile(cls, fileobj, offset, size):
    # Open underlying file
    if fileobj._compress_type != zipfile.ZIP_STORED:
        raise ValueError("Can only map uncompressed elements of a zip file")
    if fileobj._decrypter is not None:
        raise ValueError("Cannot map from an encrypted zip file")

    if size is None:
        size = fileobj._compress_size - offset
    else:
        size = min(size, fileobj._compress_size - offset)
    offset += fileobj._fileobj.tell()

    return cls.map_file(fileobj._fileobj, offset, size)

class _ZipMapBase(object):
    @classmethod
    def map_zipfile(cls, fileobj, offset = 0, size = None):
        return _map_zipfile(cls, fileobj, offset, size)

@cython.cclass
class _CZipMapBase(object):
    @classmethod
    def map_zipfile(cls, fileobj, offset = 0, size = None):
        return _map_zipfile(cls, fileobj, offset, size)

class MappedArrayProxyBase(_ZipMapBase):
    _CURRENT_VERSION = 2
    _CURRENT_MINIMUM_READER_VERSION = 2

    # Must subclass to select a schema and proxy class for writing buffers
    # Reading version-2 and above doesn't require subclassing
    schema = None
    proxy_class = None

    _Header = struct.Struct("=QQQ")
    _NewHeader = struct.Struct("=QQQQ")

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

        if self.index_elements > 0 and self.index[0] >= (self._Header.size + self._NewHeader.size):
            # New version, most likely
            self.version, min_reader_version, self.schema_offset, self.schema_size = self._NewHeader.unpack_from(
                buf, self._Header.size)
            if self._CURRENT_VERSION < min_reader_version:
                raise ValueError((
                    "Incompatible buffer, this buffer needs a reader with support for version %d at least, "
                    "this reader supports up to version %d") % (
                        min_reader_version,
                        self._CURRENT_VERSION
                    ))
            if self.schema_offset and self.schema_size:
                if self.schema_offset > len(buf) or (self.schema_size + self.schema_offset) > len(buf):
                    raise ValueError("Corrupted input - bad schema location")
                stored_schema = cPickle.loads(bytes(buffer(buf, self.schema_offset, self.schema_size)))
                if not isinstance(stored_schema, Schema):
                    raise ValueError("Corrupted input - unrecognizable schema")
                if self.schema is None or not self.schema.compatible(stored_schema):
                    self.schema = stored_schema
            elif self.schema is None:
                raise ValueError("Cannot map schema-less buffer without specifying schema")
        elif self.index_elements > 0:
            raise ValueError("Cannot reliably map version-0 buffers")

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
    @cython.locals(schema = Schema, data_pos = cython.size_t, initial_pos = cython.size_t, current_pos = object,
        schema_pos = cython.size_t, schema_end = cython.size_t)
    def build(cls, initializer, destfile = None, tempdir = None, idmap = None, return_mapper=True):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        initial_pos = destfile.tell()
        write = destfile.write
        write(cls._Header.pack(0, 0, 0))
        write(cls._NewHeader.pack(cls._CURRENT_VERSION, cls._CURRENT_MINIMUM_READER_VERSION, 0, 0))
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
        if len(index_parts) > 1:
            index = numpy.concatenate(index_parts)
        elif index_parts:
            index = index_parts[0]
        else:
            index = numpy.array([], dtype = numpy.uint64)
        del index_parts
        write(buffer(index))
        destfile.flush()

        schema_pos = destfile.tell()
        cPickle.dump(schema, destfile, 2)
        destfile.flush()

        final_pos = destfile.tell()
        destfile.seek(initial_pos)
        write(cls._Header.pack(final_pos - initial_pos, index_pos - initial_pos, len(index)))
        write(cls._NewHeader.pack(
            cls._CURRENT_VERSION, cls._CURRENT_MINIMUM_READER_VERSION,
            schema_pos - initial_pos, final_pos - schema_pos))
        destfile.flush()
        destfile.seek(final_pos)

        if return_mapper:
            return cls.map_file(destfile, initial_pos)
        else:
            return final_pos

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None):
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size)

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
        hkey = cython.longlong,
        mkey = cython.longlong,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_i64(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        if lo < hi:
            # First iteration a quick guess assuming uniform distribution of keys
            mid = min(hint, hi-1)
            mkey = cython.cast(cython.p_longlong, pindex + stride0 * mid)[0]
            if mkey < hkey:
                # Got a lo guess, now skip-search forward for a hi
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_longlong, pindex + stride0 * (mid+skip))[0] < hkey:
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
                    if cython.cast(cython.p_longlong, pindex + stride0 * (mid-skip))[0] > hkey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                # hit, but must find the first
                # good idea to go sequential because we assume collisions are unlikely
                while mid > lo and cython.cast(cython.p_longlong, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        # Final stretch: search the remaining range with a regular binary search
        while lo < hi:
            mid = (lo+hi)//2
            mkey = cython.cast(cython.p_longlong, pindex + stride0 * mid)[0]
            if mkey < hkey:
                lo = mid+1
            elif mkey > hkey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_longlong, pindex + stride0 * (mid-1))[0] == hkey:
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

    @cython.cfunc
    @cython.locals(
        hkey = cython.longlong,
        mkey = cython.longlong, ikey = cython.int, imkey = cython.int,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_i32(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        ikey = hkey
        if lo < hi:
            mid = min(hint, hi-1)
            imkey = cython.cast(cython.p_int, pindex + stride0 * mid)[0]
            if imkey < ikey:
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_int, pindex + stride0 * (mid+skip))[0] < ikey:
                        lo = mid+1
                        mid += skip
                        skip *= 2
                    else:
                        hi = mid + skip
                        break
            elif imkey > ikey:
                hi = mid
                skip = 32
                while skip > 0 and mid > lo + skip:
                    if cython.cast(cython.p_int, pindex + stride0 * (mid-skip))[0] > ikey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                while mid > lo and cython.cast(cython.p_int, pindex + stride0 * (mid-1))[0] == ikey:
                    mid -= 1
                return mid
        while lo < hi:
            mid = (lo+hi)//2
            imkey = cython.cast(cython.p_int, pindex + stride0 * mid)[0]
            if imkey < ikey:
                lo = mid+1
            elif imkey > ikey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_int, pindex + stride0 * (mid-1))[0] == ikey:
                    mid -= 1
                return mid
        return lo

    @cython.cfunc
    @cython.locals(
        hkey = cython.double,
        mkey = cython.double,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_f64(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        if lo < hi:
            # First iteration a quick guess assuming uniform distribution of keys
            mid = min(hint, hi-1)
            mkey = cython.cast(cython.p_double, pindex + stride0 * mid)[0]
            if mkey < hkey:
                # Got a lo guess, now skip-search forward for a hi
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_double, pindex + stride0 * (mid+skip))[0] < hkey:
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
                    if cython.cast(cython.p_double, pindex + stride0 * (mid-skip))[0] > hkey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                # hit, but must find the first
                # good idea to go sequential because we assume collisions are unlikely
                while mid > lo and cython.cast(cython.p_double, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        # Final stretch: search the remaining range with a regular binary search
        while lo < hi:
            mid = (lo+hi)//2
            mkey = cython.cast(cython.p_double, pindex + stride0 * mid)[0]
            if mkey < hkey:
                lo = mid+1
            elif mkey > hkey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_double, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        return lo

    @cython.cfunc
    @cython.locals(
        hkey = cython.float,
        mkey = cython.float,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t)
    @cython.returns(cython.size_t)
    def _c_search_hkey_f32(hkey, pindex, stride0, length, hint):
        hi = length
        lo = 0
        if lo < hi:
            # First iteration a quick guess assuming uniform distribution of keys
            mid = min(hint, hi-1)
            mkey = cython.cast(cython.p_float, pindex + stride0 * mid)[0]
            if mkey < hkey:
                # Got a lo guess, now skip-search forward for a hi
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast(cython.p_float, pindex + stride0 * (mid+skip))[0] < hkey:
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
                    if cython.cast(cython.p_float, pindex + stride0 * (mid-skip))[0] > hkey:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        lo = mid - skip
                        break
            else:
                # hit, but must find the first
                # good idea to go sequential because we assume collisions are unlikely
                while mid > lo and cython.cast(cython.p_float, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        # Final stretch: search the remaining range with a regular binary search
        while lo < hi:
            mid = (lo+hi)//2
            mkey = cython.cast(cython.p_float, pindex + stride0 * mid)[0]
            if mkey < hkey:
                lo = mid+1
            elif mkey > hkey:
                hi = mid
            else:
                while mid > lo and cython.cast(cython.p_float, pindex + stride0 * (mid-1))[0] == hkey:
                    mid -= 1
                return mid
        return lo

if cython.compiled:
    # Commented cython directives in pxd
    #@cython.ccall
    @cython.locals(
        lo = cython.size_t, hi = cython.size_t, hint = cython.size_t, stride0 = cython.size_t,
        indexbuf = 'Py_buffer', pindex = cython.p_char)
    #@cython.returns(cython.size_t)
    def hinted_bsearch(a, hkey, hint):
        hi = len(a)
        lo = 0
        if hi <= lo:
            return lo
        elif hkey < a[0]:
            return lo
        elif hkey > a[hi-1]:
            return hi

        #lint:disable
        PyObject_GetBuffer(a, cython.address(indexbuf), PyBUF_STRIDED_RO)
        try:
            if ( indexbuf.strides == cython.NULL
                    or indexbuf.len < hi * indexbuf.strides[0] ):
                raise ValueError("Invalid buffer state")
            pindex = cython.cast(cython.p_char, indexbuf.buf)
            stride0 = indexbuf.strides[0]
            #lint:enable
            dtype = cython.cast('char*', a.dtype.char)[0]
            if dtype == 'L' or dtype == 'Q':
                # TO-DO: better hints?
                return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint)
            elif dtype == 'I':
                # TO-DO: better hints?
                return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint)
            elif dtype == 'l' or dtype == 'q':
                # TO-DO: better hints?
                return _c_search_hkey_i64(hkey, pindex, stride0, hi, hint)
            elif dtype == 'i':
                # TO-DO: better hints?
                return _c_search_hkey_i32(hkey, pindex, stride0, hi, hint)
            elif dtype == 'd':
                # TO-DO: better hints?
                return _c_search_hkey_f64(hkey, pindex, stride0, hi, hint)
            elif dtype == 'f':
                # TO-DO: better hints?
                return _c_search_hkey_f32(hkey, pindex, stride0, hi, hint)
            else:
                raise NotImplementedError("Unsupported array type %s" % (chr(dtype),))
        finally:
            PyBuffer_Release(cython.address(indexbuf)) #lint:ok
else:
    import bisect
    def _hinted_bsearch(a, hkey, hint):
        return bisect.bisect_left(a, hkey)
    globals()['hinted_bsearch'] = _hinted_bsearch

#@cython.ccall
@cython.locals(lo = cython.size_t, hi = cython.size_t)
#@cython.returns(cython.size_t)
def bsearch(a, hkey):
    hi = len(a)
    lo = 0
    return hinted_bsearch(a, hkey, (lo+hi)//2)

#@cython.ccall
@cython.locals(lo = cython.size_t, hi = cython.size_t, ix = cython.size_t, hint = cython.size_t)
#@cython.returns(cython.bint)
def hinted_sorted_contains(a, hkey, hint):
    hi = len(a)
    ix = hinted_bsearch(a, hkey, hint)
    if ix >= hi:
        return False
    else:
        return a[ix] == hkey

#@cython.ccall
@cython.locals(lo = cython.size_t, hi = cython.size_t)
#@cython.returns(cython.bint)
def sorted_contains(a, hkey):
    hi = len(a)
    lo = 0
    return hinted_sorted_contains(a, hkey, (lo+hi)//2)

if cython.compiled:
    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.ulonglong,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_ulonglong, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_ulonglong, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_ulonglong, pdest)[0] = cython.cast(cython.p_ulonglong, pindex1)[0]
                cython.cast(cython.p_ulonglong, pdest)[1] = cython.cast(cython.p_ulonglong, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_ulonglong, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_ulonglong, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_ulonglong, pdest)[0] = cython.cast(cython.p_ulonglong, pindex2)[0]
                    cython.cast(cython.p_ulonglong, pdest)[1] = cython.cast(cython.p_ulonglong, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_ulonglong, pdest)[0] = cython.cast(cython.p_ulonglong, pindex1)[0]
            cython.cast(cython.p_ulonglong, pdest)[1] = cython.cast(cython.p_ulonglong, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_ulonglong, pdest)[0] = cython.cast(cython.p_ulonglong, pindex2)[0]
            cython.cast(cython.p_ulonglong, pdest)[1] = cython.cast(cython.p_ulonglong, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.longlong,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_longlong, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_longlong, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_longlong, pdest)[0] = cython.cast(cython.p_longlong, pindex1)[0]
                cython.cast(cython.p_longlong, pdest)[1] = cython.cast(cython.p_longlong, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_longlong, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_longlong, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_longlong, pdest)[0] = cython.cast(cython.p_longlong, pindex2)[0]
                    cython.cast(cython.p_longlong, pdest)[1] = cython.cast(cython.p_longlong, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_longlong, pdest)[0] = cython.cast(cython.p_longlong, pindex1)[0]
            cython.cast(cython.p_longlong, pdest)[1] = cython.cast(cython.p_longlong, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_longlong, pdest)[0] = cython.cast(cython.p_longlong, pindex2)[0]
            cython.cast(cython.p_longlong, pdest)[1] = cython.cast(cython.p_longlong, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.uint,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_uint, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_uint, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_uint, pdest)[0] = cython.cast(cython.p_uint, pindex1)[0]
                cython.cast(cython.p_uint, pdest)[1] = cython.cast(cython.p_uint, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_uint, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_uint, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_uint, pdest)[0] = cython.cast(cython.p_uint, pindex2)[0]
                    cython.cast(cython.p_uint, pdest)[1] = cython.cast(cython.p_uint, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_uint, pdest)[0] = cython.cast(cython.p_uint, pindex1)[0]
            cython.cast(cython.p_uint, pdest)[1] = cython.cast(cython.p_uint, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_uint, pdest)[0] = cython.cast(cython.p_uint, pindex2)[0]
            cython.cast(cython.p_uint, pdest)[1] = cython.cast(cython.p_uint, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_int, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_int, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_int, pdest)[0] = cython.cast(cython.p_int, pindex1)[0]
                cython.cast(cython.p_int, pdest)[1] = cython.cast(cython.p_int, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_int, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_int, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_int, pdest)[0] = cython.cast(cython.p_int, pindex2)[0]
                    cython.cast(cython.p_int, pdest)[1] = cython.cast(cython.p_int, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_int, pdest)[0] = cython.cast(cython.p_int, pindex1)[0]
            cython.cast(cython.p_int, pdest)[1] = cython.cast(cython.p_int, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_int, pdest)[0] = cython.cast(cython.p_int, pindex2)[0]
            cython.cast(cython.p_int, pdest)[1] = cython.cast(cython.p_int, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.double,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_f64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_double, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_double, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_double, pdest)[0] = cython.cast(cython.p_double, pindex1)[0]
                cython.cast(cython.p_double, pdest)[1] = cython.cast(cython.p_double, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_double, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_double, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_double, pdest)[0] = cython.cast(cython.p_double, pindex2)[0]
                    cython.cast(cython.p_double, pdest)[1] = cython.cast(cython.p_double, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_double, pdest)[0] = cython.cast(cython.p_double, pindex1)[0]
            cython.cast(cython.p_double, pdest)[1] = cython.cast(cython.p_double, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_double, pdest)[0] = cython.cast(cython.p_double, pindex2)[0]
            cython.cast(cython.p_double, pdest)[1] = cython.cast(cython.p_double, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.float,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_f32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast(cython.p_float, pindex2)[0]
            while pindex1 < pend1 and cython.cast(cython.p_float, pindex1)[0] <= ref and pdest < pdestend:
                cython.cast(cython.p_float, pdest)[0] = cython.cast(cython.p_float, pindex1)[0]
                cython.cast(cython.p_float, pdest)[1] = cython.cast(cython.p_float, pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast(cython.p_float, pindex1)[0]
                while pindex2 < pend2 and cython.cast(cython.p_float, pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast(cython.p_float, pdest)[0] = cython.cast(cython.p_float, pindex2)[0]
                    cython.cast(cython.p_float, pdest)[1] = cython.cast(cython.p_float, pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast(cython.p_float, pdest)[0] = cython.cast(cython.p_float, pindex1)[0]
            cython.cast(cython.p_float, pdest)[1] = cython.cast(cython.p_float, pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast(cython.p_float, pdest)[0] = cython.cast(cython.p_float, pindex2)[0]
            cython.cast(cython.p_float, pdest)[1] = cython.cast(cython.p_float, pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) / stride0

    # Commented cython directives in pxd
    #@cython.ccall
    @cython.locals(
        stride0 = cython.size_t, length1 = cython.size_t, length2 = cython.size_t,
        destlength = cython.size_t, rdestlength = cython.size_t,
        index1buf = 'Py_buffer', index2buf = 'Py_buffer', destbuf = 'Py_buffer',
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def index_merge(index1, index2, dest):
        length1 = len(index1)
        length2 = len(index2)
        destlength = len(dest)
        rdestlength = min(destlength, length1+length2)

        if len(index1.shape) != 2 or len(index2.shape) != 2:
            raise ValueError("Indices must be two-dimensional")
        if len(dest.shape) != 2:
            raise ValueError("Destination must be two-dimensional")
        if index1.shape[1] != 2 or index2.shape[1] != 2:
            raise ValueError("Indices must be N x 2 matrices")
        if dest.shape[1] != 2:
            raise ValueError("Destination must be an N x 2 matrix")

        #lint:disable
        PyObject_GetBuffer(index1, cython.address(index1buf), PyBUF_STRIDED_RO)
        try:
            PyObject_GetBuffer(index2, cython.address(index2buf), PyBUF_STRIDED_RO)
            try:
                PyObject_GetBuffer(dest, cython.address(destbuf), PyBUF_STRIDED_RO)
                try:
                    if ( index1buf.strides == cython.NULL
                            or index1buf.len < length1 * index1buf.strides[0] ):
                        raise ValueError("Invalid buffer state on index1")
                    if ( index2buf.strides == cython.NULL
                            or index2buf.len < length2 * index2buf.strides[0] ):
                        raise ValueError("Invalid buffer state on index2")
                    if ( destbuf.strides == cython.NULL
                            or destbuf.len < destlength * destbuf.strides[0] ):
                        raise ValueError("Invalid buffer state on dest")

                    pindex1 = cython.cast(cython.p_char, index1buf.buf)
                    pindex2 = cython.cast(cython.p_char, index2buf.buf)
                    pdest = cython.cast(cython.p_char, destbuf.buf)
                    stride0 = index1buf.strides[0]
                    if stride0 == 0:
                        raise ValueError("Invalid buffer stride")
                    if index2buf.strides[0] != stride0:
                        raise ValueError("Incompatible indexes")
                    if destbuf.strides[0] != stride0:
                        raise ValueError("Incompatible destination")

                    dtype = cython.cast('char*', index1.dtype.char)[0]
                    if cython.cast('char*', index2.dtype.char)[0] != dtype:
                        raise ValueError("Incompatible indexes")
                    if cython.cast('char*', dest.dtype.char)[0] != dtype:
                        raise ValueError("Incompatible destination")

                    if ( pdest == pindex1 or pdest == pindex2
                            or (pdest < pindex1 + length1 * stride0 and pdest >= pindex1)
                            or (pdest < pindex2 + length2 * stride0 and pdest >= pindex2)
                            or (pindex1 < pdest + rdestlength * stride0 and pindex1 >= pdest)
                            or (pindex2 < pdest + rdestlength * stride0 and pindex2 >= pdest) ):
                        raise NotImplementedError("In-place merge not implemented, destination must not overlap")

                    #lint:enable
                    if dtype == 'L' or dtype == 'Q':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_ui64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'I':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_ui32(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'l' or dtype == 'q':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_i64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'i':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_i32(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'd':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_f64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'f':
                        # TO-DO: better hints?
                        with cython.nogil:
                            return _c_merge_f32(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    else:
                        raise NotImplementedError("Unsupported array type %s" % (chr(dtype),))
                finally:
                    PyBuffer_Release(cython.address(destbuf)) #lint:ok
            finally:
                PyBuffer_Release(cython.address(index2buf)) #lint:ok
        finally:
            PyBuffer_Release(cython.address(index1buf)) #lint:ok
else:
    # Not so efficient pure-python fallback
    def _index_merge(index1, index2, dest):
        # Main merge
        pdest = 0
        pindex1 = 0
        pindex2 = 0
        pend1 = len(index1)
        pend2 = len(index2)
        pdestend = len(dest)

        if len(index1.shape) != 2 or len(index2.shape) != 2:
            raise ValueError("Indices must be two-dimensional")
        if len(dest.shape) != 2:
            raise ValueError("Destination must be two-dimensional")
        if index1.shape[1] != 2 or index2.shape[1] != 2:
            raise ValueError("Indices must be N x 2 matrices")
        if dest.shape[1] != 2:
            raise ValueError("Destination must be an N x 2 matrix")
        if (    (dest.base if dest.base is not None else dest) is (
                    index1.base if index1.base is not None else index1)
                or (dest.base if dest.base is not None else dest) is (
                    index2.base if index2.base is not None else index2) ):
            raise NotImplementedError("In-place merge not implemented, destination must not overlap")

        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = index2[pindex2][0]
            while pindex1 < pend1 and index1[pindex1][0] <= ref and pdest < pdestend:
                dest[pdest] = index1[pindex1]
                pdest += 1
                pindex1 += 1
            if pindex1 < pend1:
                ref = index1[pindex1][0]
                while pindex2 < pend2 and index2[pindex2][0] <= ref and pdest < pdestend:
                    dest[pdest] = index2[pindex2]
                    pdest += 1
                    pindex2 += 1

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            dest[pdest] = index1[pindex1]
            pdest += 1
            pindex1 += 1
        while pindex2 < pend2 and pdest < pdestend:
            dest[pdest] = index2[pindex2]
            pdest += 1
            pindex2 += 1
        return pdest
    globals()['index_merge'] = _index_merge

@cython.ccall
@cython.locals(i = int, clobber = cython.bint)
def _merge_all(parts, dtype, clobber):
    if len(parts) == 1:
        return parts[0]
    else:
        nparts = []
        for i in xrange(0, len(parts), 2):
            if i+1 < len(parts):
                npart = numpy.empty((len(parts[i])+len(parts[i+1]), 2), dtype)
                merge_elements = index_merge(parts[i], parts[i+1], npart)
                if merge_elements != len(npart):
                    npart = npart[:merge_elements]
                nparts.append(npart)
            else:
                nparts.append(parts[i])
        if clobber:
            del parts[:]
        del parts
        return _merge_all(nparts, dtype, True)

@cython.cfunc
@cython.locals(discard_duplicate_keys = cython.bint, discard_duplicates = cython.bint, copy = cython.bint,
    nout = cython.size_t, nmin = cython.size_t, out_start = cython.size_t)
def _discard_duplicates(apart, struct_dt, discard_duplicate_keys, discard_duplicates, copy):
    if discard_duplicate_keys or discard_duplicates:
        # What numpy.unique does, but using MUCH less RAM
        vpart = apart.view(struct_dt)
        vpart.sort(0)
        if discard_duplicate_keys:
            flags = apart[1:,0] != apart[:-1,0]
        elif discard_duplicates:
            flags = vpart[1:,0] != vpart[:-1,0]
        np = numpy
        if not np.all(flags):
            if copy:
                # Simple boolean indexing, best implementation for constructing a deduplicated copy
                flags = np.concatenate([[True], flags])
                apart = apart[flags]
            else:
                # In-place compress operation, best option for non-copying deduplication, to avoid
                # having to allocate temporary workspace proportional to output size. This
                # way only uses a fixed amount of temporary workspace and builds a compressed
                # result into the input array directly.
                nmin = np.argmin(flags)
                count_nonzero = np.count_nonzero
                start = nmin
                end = len(flags)
                out_start = start+1
                while start < end:
                    chunk_end = min(end, start + 100000)
                    flags_slice = flags[start:chunk_end]
                    nout = count_nonzero(flags_slice)
                    if nout != 0:
                        dedup_slice = apart[start+1:chunk_end+1]
                        if nout < (chunk_end - start):
                            dedup_slice = dedup_slice[flags_slice]
                        apart[out_start:out_start+nout] = dedup_slice
                        del dedup_slice
                    del flags_slice
                    start = chunk_end
                    out_start += nout
                apart = apart[:out_start]
    return apart

@cython.cclass
class NumericIdMapper(_CZipMapBase):
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

    @property
    def fileobj(self):
        return self._file

    @property
    def fileno(self):
        return self._file.fileno()

    @property
    def name(self):
        return self._file.name

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
            try:
                self.dtypemax = ~dtype.type(0)
            except:
                self.dtypemax = ~0

    def __getitem__(self, key):
        rv = self.get(key)
        if rv is None:
            raise KeyError(key)
        else:
            return rv

    def __len__(self):
        return self.index_elements

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
        else:
            dtype = self.dtype
            struct_dt = numpy.dtype([
                ('key', dtype),
                ('value', dtype),
            ])
            return self.index.view(struct_dt).reshape(self.index.shape[0]).searchsorted(
                numpy.array([(hkey,0)],dtype=struct_dt))[0]

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
            discard_duplicates = False, discard_duplicate_keys = False, return_mapper=True):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)
        partsfile = partswrite = None

        try:
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
            bigparts = []
            parts = []
            islice = itertools.islice
            array = numpy.array
            curpos = basepos + cls._Header.size
            part = []
            struct_dt = numpy.dtype([
                ('key', dtype),
                ('value', dtype),
            ])
            void_dt = numpy.dtype((numpy.void, struct_dt.itemsize))
            concatenate = numpy.concatenate
            while 1:
                del part[:]
                for k,i in islice(initializer, 1000):
                    # Add the index item
                    part.append((k,i))
                if part:
                    parts.append(_discard_duplicates(
                        array(part, dtype), void_dt,
                        discard_duplicate_keys, discard_duplicates, False))
                else:
                    break
                if len(parts) > 1000:
                    # merge into a big part to flatten
                    apart = concatenate(parts)
                    del parts[:]
                    apart = _discard_duplicates(
                        apart, void_dt,
                        discard_duplicate_keys, discard_duplicates,
                        tempdir is None)
                    if tempdir is not None:
                        # Accumulate in tempfile
                        if partsfile is None:
                            partsfile = tempfile.TemporaryFile(dir = tempdir)
                            partswrite = partsfile.write
                        partswrite(buffer(apart))
                    else:
                        # Accumulate in memory
                        bigparts.append(apart)
                    del apart
            del part

            bigparts.extend(parts)
            del parts

            if partsfile is not None:
                if bigparts:
                    # Flush the rest to do the final sort in mapped memory
                    for apart in bigparts:
                        partswrite(buffer(apart))
                    del bigparts[:]
                partsfile.flush()
                partsfile.seek(0)
                apart = numpy.memmap(partsfile, dtype).reshape(-1,2)
                apart = _discard_duplicates(
                    apart, struct_dt,
                    discard_duplicate_keys, discard_duplicates, False)
                bigparts.append(apart)
                del apart

            # Merge the final batch of parts and build the sorted index
            if bigparts:
                needs_resort = not (discard_duplicate_keys or discard_duplicates)
                if len(bigparts) > 1:
                    bigindex = concatenate(bigparts)
                    del bigparts[:]
                    bigindex = _discard_duplicates(
                        bigindex, struct_dt,
                        discard_duplicate_keys, discard_duplicates, False)
                else:
                    bigindex = bigparts[0]
                    del bigparts[:]
                    if partsfile is None:
                        # Must re-sort with structural sort. Void sort may not have the same ordering.
                        needs_resort = True
                if needs_resort:
                    # Just sort, else already deduplicated and sorted
                    bigindex.view(struct_dt).sort(0)
                index = bigindex
                del bigindex
            else:
                index = numpy.empty(shape=(0,2), dtype=dtype)
            del bigparts

            indexpos = curpos
            write(buffer(index))
            nitems = len(index)
        finally:
            if partsfile is not None:
                partsfile.close()
            partsfile = partswrite = None

        finalpos = destfile.tell()
        if finalpos & 31:
            write("\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'NumericIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None):
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size)

        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        fileobj.seek(map_start)
        buf = mmap.mmap(fileobj.fileno(), 0, access = mmap.ACCESS_READ, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

    @classmethod
    @cython.locals(
        basepos = cython.ulonglong, curpos = cython.ulonglong, endpos = cython.ulonglong, finalpos = cython.ulonglong,
        discard_duplicates = cython.bint, discard_duplicate_keys = cython.bint, index_elements = cython.size_t,
        mapper = 'NumericIdMapper')
    def merge(cls, parts, destfile = None, tempdir = None,
            discard_duplicates = False, discard_duplicate_keys = False):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        basepos = destfile.tell()

        # Reserve space for the header
        write = destfile.write
        write(cls._Header.pack(0, 0))

        indexpos = basepos + cls._Header.size

        # Merge the indexes
        index = _merge_all([mapper.index for mapper in parts], dtype, True)

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

class NumericId32Mapper(NumericIdMapper):
    dtype = npuint32

@cython.cclass
class ObjectIdMapper(_CZipMapBase):
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
    )

    @property
    def buf(self):
        return self._buf

    @property
    def fileobj(self):
        return self._file

    @property
    def fileno(self):
        return self._file.fileno()

    @property
    def name(self):
        return self._file.name

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

        self.index = numpy.ndarray(
            buffer = self._buf,
            offset = self.index_offset,
            dtype = self.dtype,
            shape = (self.index_elements, 3))

    def __getitem__(self, key):
        rv = self.get(key)
        if rv is None:
            raise KeyError(key)
        else:
            return rv

    def __len__(self):
        return self.index_elements

    def __contains__(self, key):
        return self.get(key) is not None

    def preload(self):
        # Just touch everything in sequential order
        self.index.max()

    @cython.ccall
    def _unpack(self, buf, index):
        return mapped_object.unpack_from(buf, index)

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer', pybuf = 'Py_buffer')
    def iterkeys(self, make_sequential = False):
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
                            yield self._unpack(self._buf,
                                cython.cast(cython.p_ulonglong, indexbuf.buf)[i*stride+offs])
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                elif dtype is npuint32:
                    PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_SIMPLE)
                    try:
                        if indexbuf.len < (self.index_elements * stride * cython.sizeof(cython.uint)):
                            raise ValueError("Invalid buffer state")
                        for i in xrange(self.index_elements):
                            yield self._unpack(self._buf,
                                cython.cast(cython.p_uint, indexbuf.buf)[i*stride+offs])
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in xrange(self.index_elements):
                        yield self._unpack(self._buf, index[i])
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in xrange(self.index_elements):
                yield self._unpack(buf, index[i])

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
    def iteritems(self, make_sequential = False):
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
                                self._unpack(self._buf,
                                    cython.cast(cython.p_ulonglong, pindex + stride1)[0]),
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
                                self._unpack(self._buf,
                                    cython.cast(cython.p_uint, pindex + stride1)[0]),
                                cython.cast(cython.p_uint, pindex + 2*stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in xrange(self.index_elements):
                        yield (
                            self._unpack(self._buf, index[i,1]),
                            index[i,2]
                        )
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in xrange(self.index_elements):
                yield (self._unpack(buf, index[i,1]), index[i,2])

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
        else:
            dtype = self.dtype
            struct_dt = numpy.dtype([
                ('key_hash', dtype),
                ('key_offset', dtype),
                ('value', dtype),
            ])
            return self.index.view(struct_dt).reshape(self.index.shape[0]).searchsorted(
                numpy.array([(hkey,0,0)],dtype=struct_dt))[0]

    @cython.ccall
    def _compare_keys(self, buf, offs, key):
        return self._unpack(buf, offs) == key

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int,
        stride0 = cython.size_t, stride1 = cython.size_t,
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get(self, key, default = None):
        if not isinstance(key, basestring):
            return default
        hkey = _stable_hash(key)
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
                            while pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                if self._compare_keys(self._buf,
                                      cython.cast(cython.p_ulonglong, pindex + stride1)[0],
                                      key):
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
                                if self._compare_keys(self._buf,
                                         cython.cast(cython.p_uint, pindex + stride1)[0],
                                         key):
                                    return cython.cast(cython.p_uint, pindex + 2*stride1)[0]
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            if self._compare_keys(self._buf, self.index[startpos,1], key):
                                return index[startpos,2]
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    if self._compare_keys(buf, index[startpos, 1], key):
                        return index[startpos,2]
                    startpos += 1
        return default

    @classmethod
    @cython.locals(
        basepos = cython.ulonglong, curpos = cython.ulonglong, endpos = cython.ulonglong, finalpos = cython.ulonglong,
        dtypemax = cython.ulonglong)
    def build(cls, initializer, destfile = None, tempdir = None, return_mapper=True, min_buf_size=128):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        try:
            dtypemax = ~dtype(0)
        except:
            try:
                dtypemax = ~dtype.type(0)
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
        islice = itertools.islice
        array = numpy.array
        curpos = basepos + cls._Header.size
        pack_into = mapped_object.pack_into
        valbuf = bytearray(65536)
        valbuflen = 65536
        n = 0
        part = []
        while 1:
            del part[:]
            for k,i in islice(initializer, 1000):
                # Add the index item
                n += 1
                # Not *quite* as correct as computing the actual length in
                # bytes, but this is only used as a first (optimistic) estimation.
                klen = max(min_buf_size, sys.getsizeof(k))
                if curpos > dtypemax:
                    raise ValueError("Cannot represent offset with requested precision")
                part.append((_stable_hash(k), curpos, i))

                while True:
                    try:
                        if klen + 16 > valbuflen:
                            valbuflen = (klen + 16) * 2
                            valbuf = bytearray(valbuflen)
                        endpos = pack_into(k, valbuf, 0)
                        break
                    except (struct.error, IndexError):
                        klen += klen

                if valbuflen < endpos:
                    raise RuntimeError("Buffer overflow")
                write(valbuf[:endpos])
                curpos += endpos
            if part:
                parts.append(array(part, dtype))
            else:
                break
        if parts:
            index = numpy.concatenate(parts)
        else:
            index = numpy.empty(shape=(0,3), dtype=dtype)
        del parts, part
        shuffle = numpy.argsort(index[:,0])
        index = index[shuffle]

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

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'ObjectIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None):
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size)

        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        fileobj.seek(map_start)
        buf = mmap.mmap(fileobj.fileno(), 0, access = mmap.ACCESS_READ, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

@cython.locals(ux = unicode)
def safe_utf8(x):
    if isinstance(x, unicode):
        # The assignment-style cast is needed because an inline case triggers a Cython compiler crash
        ux = x
        return ux.encode("utf8")
    else:
        return x

@cython.cclass
class StringIdMapper(_CZipMapBase):
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

    @property
    def fileobj(self):
        return self._file

    @property
    def fileno(self):
        return self._file.fileno()

    @property
    def name(self):
        return self._file.name

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

    def __len__(self):
        return self.index_elements

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
        else:
            dtype = self.dtype
            struct_dt = numpy.dtype([
                ('key_hash', dtype),
                ('key_offset', dtype),
                ('value', dtype),
            ])
            return self.index.view(struct_dt).reshape(self.index.shape[0]).searchsorted(
                numpy.array([(hkey,0,0)],dtype=struct_dt))[0]

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
    def build(cls, initializer, destfile = None, tempdir = None, return_mapper=True):
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        try:
            dtypemax = ~dtype(0)
        except:
            try:
                dtypemax = ~dtype.type(0)
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
            index = numpy.concatenate(parts)
        else:
            index = numpy.empty(shape=(0,3), dtype=dtype)
        del parts, part
        shuffle = numpy.argsort(index[:,0])
        index = index[shuffle]

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

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'StringIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None):
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size)

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
        hkey = cython.ulonglong, startpos = int, nitems = int,
        stride0 = cython.size_t, stride1 = cython.size_t,
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

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int,
        stride0 = cython.size_t,
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def __contains__(self, key):
        if not isinstance(key, (int, long)):
            return False
        if key < 0 or key > self.dtypemax:
            return False
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
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            if pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                return True
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
                            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
                            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
                            if pindex < pindexend and cython.cast(cython.p_uint, pindex)[0] == hkey:
                                return True
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        if startpos < nitems and index[startpos,0] == hkey:
                            return True
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                if startpos < nitems and index[startpos,0] == hkey:
                    return True
        return False

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int,
        stride0 = cython.size_t, stride1 = cython.size_t,
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get_iter(self, key):
        if not isinstance(key, (int, long)):
            return
        if key < 0 or key > self.dtypemax:
            return
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
                            while pindex < pindexend and cython.cast(cython.p_ulonglong, pindex)[0] == hkey:
                                yield cython.cast(cython.p_ulonglong, pindex + stride1)[0]
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
                                yield cython.cast(cython.p_uint, pindex + stride1)[0]
                                pindex += stride0
                        finally:
                            PyBuffer_Release(cython.address(indexbuf))
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            yield index[startpos,1]
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    yield index[startpos,1]
                    startpos += 1

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

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get_iter(self, key, default = None):
        if not isinstance(key, basestring):
            return
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
                                    yield cython.cast(cython.p_ulonglong, pindex + 2*stride1)[0]
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
                                    yield cython.cast(cython.p_uint, pindex + 2*stride1)[0]
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
                                yield index[startpos,2]
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    if _unpack_bytes_from_pybuffer(buf, index[startpos,1], None) == bkey:
                        yield index[startpos,2]
                    startpos += 1

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def __contains__(self, key):
        if not isinstance(key, basestring):
            return False
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
                                    return True
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
                                    return True
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
                                return True
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    if _unpack_bytes_from_pybuffer(buf, index[startpos,1], None) == bkey:
                        return True
                    startpos += 1
        return False

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
        if isinstance(key, (int, long)):
            return super(ApproxStringIdMultiMapper, self).get(key, default)
        else:
            return super(ApproxStringIdMultiMapper, self).get(self._xxh(self._encode(key)).intdigest(), default)

    def __contains__(self, key):
        if isinstance(key, (int, long)):
            return super(ApproxStringIdMultiMapper, self).__contains__(key)
        else:
            return super(ApproxStringIdMultiMapper, self).__contains__(self._xxh(self._encode(key)).intdigest())

    def get_iter(self, key):
        if isinstance(key, (int, long)):
            return super(ApproxStringIdMultiMapper, self).get_iter(key)
        else:
            return super(ApproxStringIdMultiMapper, self).get_iter(self._xxh(self._encode(key)).intdigest())

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

class MappedMappingProxyBase(_ZipMapBase):
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
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size)

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

    def get_iter(self, key):
        for ix in self.id_mapper.get_iter(key):
            yield self.value_array[ix]

_cythonized = cython.compiled
