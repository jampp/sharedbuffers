# -*- coding: utf-8 -*-
# cython: infer_types=True, profile=False, linetrace=False
# distutils: define_macros=CYTHON_TRACE=0
"""
This modules implements marshalling of complex structures into shared memory buffers
and proxy classes needed to access the information within them in an efficient manner
without previous deserialization.

The :class:`Schema` class represents a particular structure's shape in shared
memory. Shared buffers can span complex structures by placing an object of a known
schema at a known location. Then :meth:`Schema.unpack_from` can be used to
get a proxy to that buffer that will expose the whole structure accessible in
pythonic (and efficient) form.

Schemas are structures with a static list of optionally present fields. All declared
fields can be either missing, None, or have a value of the declared type. The schema
defines how the shared memory will be interpreted, and it's not normally stored explicitly
in the shared memory itself, so "freestyle" classes with arbitrary attributes aren't
supported in this fashion.

Schemas can be declared with fields of any of the built-in primitive types:

Numbers
  :class:`ubyte`, :class:`byte`, :class:`ushort`, :class:`short`,
  :class:`uint32`, :class:`int32`, :class:`uint64`, :class:`int64`,
  :class:`float32`, :class:`float64`, or their python type, :class:`~decimal.Decimal`,
  :class:`int`, :class:`float`, :class:`bool`.

Dates
  :class:`datetime.datetime`, :class:`datetime.date`.

Strings
  by their python type, :class:`str`, :class:`bytes`, :class:`unicode`,
  or explicitly by their built-in implementation type :class:`mapped_bytes`, :class:`mapped_unicode`.

Buffers
  by python's :class:`buffer` or :class:`proxied_buffer`.

Containers
  by python's :class:`list`, :class:`tuple`, :class:`frozenset`, :class:`dict`,
  or by their built-in implementations, which can customize their proxying behavior, :class:`mapped_tuple`,
  :class:`mapped_list`, :class:`mapped_dict`, :class:`mapped_frozenset`, :class:`proxied_tuple`,
  :class:`proxied_list`, :class:`proxied_dict`, :class:`proxied_frozenset`.

Numpy arrays
  by declaring them as :class:`proxied_ndarray`.

Fields can also be declared of *dynamic* type, which means the value will be wrapped with
runtime type information, and it will accept any value of any supported type, by declaring
them as :class:`object` (or :class:`mapped_object` to be more explicit).

Once a :class:`Schema` is initialized, data can be placed in any writable buffer by
invoking :meth:`~Schema.pack_into`, and accessed through a proxy construced by :meth:`~Schema.unpack_from`.

In order to make fields reference objects other than the built-in types, their :class:`Schema` has to be
described and registered with :meth:`mapped_object.register_schema`, after which it can be used in field
type declarations as if it were another built-in type, and within containers or dynamically typed values.

:class:`Schema` instances are pickleable, which means they can be embedded into shared memory buffers
to make them portable. This is not done automatically except in some high level data structures,
when explicitly stated.
"""

import struct
import array
import mmap
import numpy
import tempfile
import functools
import os
import sys

import six
import xxhash
import itertools
import time
import zipfile
import math
import sys
import collections
import weakref
import ctypes
from datetime import timedelta, datetime, date
from decimal import Decimal
from six import itervalues, reraise, iteritems, iterkeys
from six.moves import cPickle, range


try:
    from cdecimal import Decimal as cDecimal
except:
    cDecimal = Decimal

try:
    from chorde.clients.inproc import Cache
except ImportError:
    from clru.lrucache import LRUCache as Cache

try:
    from chorde.clients.inproc import CuckooCache as FastCache
except ImportError:
    FastCache = Cache

import cython

npuint64 = cython.declare(object, numpy.uint64)
npint64 = cython.declare(object, numpy.int64)
npuint32 = cython.declare(object, numpy.uint32)
npint32 = cython.declare(object, numpy.int32)
npuint16 = cython.declare(object, numpy.uint16)
npint16 = cython.declare(object, numpy.int16)
npuint8 = cython.declare(object, numpy.uint8)
npint8 = cython.declare(object, numpy.int8)
npfloat64 = cython.declare(object, numpy.float64)
npfloat32 = cython.declare(object, numpy.float32)
npempty = cython.declare(object, numpy.empty)
npfrombuffer = cython.declare(object, numpy.frombuffer)
npndarray = cython.declare(object, numpy.ndarray)

frexp = cython.declare(object, math.frexp)

ctypes_Array = cython.declare(object, ctypes.Array)

if six.PY3:
    long = int
    basestring = (bytes, str)

if cython.compiled:
    # Compatibility fix for cython >= 0.23, which no longer supports "buffer" as a built-in type
    if six.PY2:
        buffer = cython.declare(object, buffer)  # lint:ok
        from types import BufferType as buffer
    else:
        buffer = memoryview

    assert Py_LT == 0
    assert Py_LE == 1
    assert Py_EQ == 2
    assert Py_NE == 3
    assert Py_GT == 4
    assert Py_GE == 5
else:
    if six.PY3:
        buffer = memoryview

def buffer_with_offset(data, offset, size):
    if six.PY3:
        return buffer(data)[offset:offset+size]
    return buffer(data, offset, size)

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
    if (type(buf) is buffer or type(buf) is bytearray or type(buf) is bytes or
            isinstance(buf, (ctypes_Array, bytes))):
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

class NONE:
    pass

@cython.cclass
class StrongIdMap(object):
    cython.declare(
        __weakref__ = object,
        preloaded = dict,
        idmap = dict,
        objmap = object,
        strong_refs = object,
        stable_set = set,
    )

    def __init__(self, strong_limit = 1 << 20 - 1, preallocate = False, strong_class = FastCache, stable_set = None):
        """
        Constructs a strong-referencing id map. The id map will keep strong references when necessary
        to guarantee correct behavior, up to ``strong_limit`` references. After that, unused references
        are evicted from both the strong-reference list and the id map itself, maintaining correct
        behavior at the expense of deduplication effectiveness.

        :param strong_limit: Keep at most this many strong references

        :param preallocate: Passed to ``strong_class`` as a kwarg

        :param strong_class: The cache constructor used for the strong reference list. By default, it uses a kind
            of LRU cache. The constructor should accept ``preallocate`` as kwarg and ``strong_limit`` as
            first positional argument. When given ``preallocate=true``, the structure should be preallocated
            to accommodate ``strong_limit`` elements. It should also accept an ``eviction_callback`` kwarg
            with a callable to be called with ``(key, value)`` arguments when evicting items from the mapping.

        :param stable_set: A set of shared_id s that are considered stable. That is, strong references are kept
            by the caller, so they don't need to be tracked. This allows them to be persistent on the id map
            and thus improve deduplication effectiveness with little overhead.
        """
        self.preloaded = {}
        self.idmap = {}
        self.strong_refs = strong_class(
            strong_limit,
            eviction_callback = functools.partial(
                self._on_strong_evict,
                weakref.ref(self),
            ),
            preallocate = preallocate,
        )
        self.objmap = weakref.WeakValueDictionary()
        if stable_set is not None and not isinstance(stable_set, set):
            stable_set = set(stable_set)
        self.stable_set = stable_set

    @staticmethod
    @cython.locals(self = 'StrongIdMap')
    def _on_strong_evict(wself, key, value):
        self = wself()
        if self is not None:
            self.pop(key, None)

    def __len__(self):
        return len(self.idmap) + len(self.preloaded)

    def __iter__(self):
        for key in self.preloaded:
            yield key
        for key in self.idmap:
            if key in self and key not in self.preloaded:
                yield key

    def iterkeys(self):
        return iter(self)

    def itervalues(self):
        for key in self:
            yield self[key]

    def iteritems(self):
        for key in self:
            yield key, self[key]

    def keys(self):
        return list(self)

    def values(self):
        return list(self.itervalues())

    def items(self):
        return list(self.iteritems())

    def __setitem__(self, key, value):
        self.idmap[key] = value
        self.objmap[key] = NONE

    def __getitem__(self, key):
        if key in self.preloaded:
            return self.preloaded[key]
        if key not in self.objmap:
            self.idmap.pop(key, None)
            raise KeyError(key)
        return self.idmap[key]

    def __delitem__(self, key):
        self.objmap.pop(key, None)
        self.strong_refs.pop(key, None)
        del self.idmap[key]

    def __contains__(self, key):
        return key in self.preloaded or (key in self.idmap and key in self.objmap)

    def preload(self, mapping):
        self.preloaded.update(mapping)

    def clear(self):
        self.objmap.clear()
        self.idmap.clear()
        self.strong_refs.clear()

    def clear_preloaded(self):
        self.preloaded.clear()

    @cython.ccall
    def get(self, key, default=None):
        if key in self.preloaded:
            return self.preloaded[key]
        if key not in self.objmap:
            self.idmap.pop(key, None)
            return default
        return self.idmap.get(key, default)

    @cython.ccall
    def pop(self, key, default=NONE):
        if key in self.preloaded:
            # No popping from preloaded items
            return self.preloaded[key]
        orv = self.objmap.pop(key, NONE)
        rv = self.idmap.pop(key, NONE)
        self.strong_refs.pop(key, None)
        if orv is NONE:
            rv = NONE
        if rv is not NONE:
            return rv
        elif default is NONE:
            raise KeyError(key)
        else:
            return default

    @cython.ccall
    @cython.returns(cython.bint)
    def link(self, key, obj):
        if isinstance(obj, (proxied_list, proxied_dict, BufferProxyObject)):
            # There's no need to link lifecycles, the object
            # is identified uniquely by its buffer mapping and that is a stable id
            return True
        if is_equality_key(key):
            # These are equality keys, they are hard references already
            return True
        if self.stable_set is not None:
            if key in self.stable_set:
                return True
            elif is_wrapped_key(key) and get_wrapped_key(key) in self.stable_set:
                return True

        try:
            self.objmap[key] = obj
            return True
        except TypeError:
            # Not weakly referenceable, try to hold a strong reference then to stop
            # its id from being reused while we hold its idmap entry
            self.strong_refs[key] = obj
            return False


@cython.ccall
@cython.inline
@cython.returns(cython.bint)
def is_equality_key(obj):
    # same as is_equality_key_compatible except for integer keys,
    # where the LONG_MASK must be checked
    if (obj is None or obj is ()
            or (isinstance(obj, (int, long)) and obj & LONG_MASK)
            or (isinstance(obj, basestring) and len(obj) < 16)):
        return True
    elif is_wrapped_key(obj):
        return is_equality_key(get_wrapped_key(obj))
    else:
        return False


@cython.ccall
@cython.inline
@cython.returns(cython.bint)
def is_equality_key_compatible(obj):
    # singletons, small strings and integers
    if (obj is None or obj is ()
            or isinstance(obj, (int, long))
            or (isinstance(obj, basestring) and len(obj) < 16)):
        return True
    elif is_wrapped_key(obj):
        return is_equality_key(get_wrapped_key(obj))
    else:
        return False


@cython.ccall
@cython.locals(lobj = 'proxied_list', dobj = 'proxied_dict', oobj = 'BufferProxyObject')
def shared_id(obj):
    # (id(buf), offs) of buffer-mapped proxies
    if isinstance(obj, proxied_list):
        lobj = obj
        rv = (id(lobj.buf) << 68) | lobj.offs | PROXY_MASK
        if lobj.elem_step != 0:
            # Add slice arguments to the shared_id
            rv |= long(lobj.elem_step) << (68 + 64)
            rv |= long(lobj.elem_start) << (68 + 128)
            rv |= long(lobj.elem_end) << (68 + 192)
        return rv
    elif isinstance(obj, proxied_dict):
        dobj = obj
        return (id(dobj.buf) << 68) | dobj.offs | PROXY_MASK
    elif isinstance(obj, BufferProxyObject):
        oobj = obj
        return (id(oobj.buf) << 68) | oobj.offs | PROXY_MASK
    elif is_equality_key_compatible(obj):
        # For numeric keys, just add the LONG_MASK to differentiate them from regular ids
        if isinstance(obj, int):
            # Up to 64 bits, just add the LONG_MASK
            return obj | LONG_MASK
        elif isinstance(obj, long):
            # Real longs must make room for the flag bits
            return (obj & 0xFFFFFFFFFFFFFFFF) | ((obj >> 64) << 68) | LONG_MASK

        # For other keys, use the object itself as key if hashable
        try:
            hash(obj)
        except TypeError:
            pass
        else:
            return obj

    # Otherwise plainly the id of the object
    return id(obj)


class WRAPPED:
    pass


WRAP_MASK = cython.declare(object, 1 << 64)
LONG_MASK = cython.declare(object, 2 << 64)
PROXY_MASK = cython.declare(object, 4 << 64)
FLAG_MASK = cython.declare(object, 0xF << 64)


@cython.ccall
def wrapped_id(obj):
    xid = shared_id(obj)
    if isinstance(xid, (int, long)):
        # We keep numeric ids numeric
        xid |= WRAP_MASK
    else:
        # Equality ids are wrapped in a tuple
        xid = (WRAPPED, xid)
    return xid


@cython.ccall
@cython.inline
@cython.returns(cython.bint)
def is_wrapped_key(obj):
    # keys for type-tagged objects
    if isinstance(obj, tuple):
        return len(obj) == 2 and obj[0] is WRAPPED
    elif isinstance(obj, (int, long)):
        return (obj & WRAP_MASK) == WRAP_MASK


@cython.ccall
@cython.inline
def get_wrapped_key(obj):
    # the key for the unwrapped value
    if isinstance(obj, tuple):
        return obj[1]
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, long):
        return obj & ~WRAP_MASK


class mapped_frozenset(frozenset):

    @classmethod
    @cython.locals(cbuf = 'unsigned char[:]', i=int, ix=int, offs=cython.longlong)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if isinstance(obj, npndarray):
            all_int = all_intlong = all_float = 0
            obj_dtype = obj.dtype
            if obj_dtype.isbuiltin:
                dtype = obj_dtype.char
                cdtype = cython.cast('const char*', dtype)[0]
                if cdtype in ('l', 'I', 'i', 'H', 'h', 'B', 'b'):
                    all_int = all_intlong = 1
                elif cdtype == 'L':
                    all_intlong = 1
                elif cdtype in ('d', 'f'):
                    all_float = 1
        else:
            all_int = all_intlong = all_float = 1
            for x in obj:
                if all_int and type(x) is not int:
                    all_int = 0
                if all_intlong and type(x) is not int and type(x) is not long:
                    all_intlong = 0
                if all_float and type(x) is not float:
                    all_float = 0
                if not all_int and not all_intlong and not all_float:
                    break
        if all_int or all_intlong:
            if isinstance(obj, npndarray):
                maxval = int(obj.max())
                minval = int(obj.min())
            else:
                maxval = max(obj) if obj else 0
                minval = min(obj) if obj else 0
            if 0 <= minval and maxval < 120:
                # inline bitmap
                try:
                    cbuf = buf
                    isbuffer = True
                except:
                    isbuffer = False
                if cython.compiled and isbuffer:
                    if maxval < 56:
                        cbuf[offs] = 'm'
                        for i in range(1, 8):
                            cbuf[offs+i] = 0
                    else:
                        cbuf[offs] = 'M'
                        for i in range(1, 16):
                            cbuf[offs+i] = 0
                    for ix in obj:
                        cbuf[offs+1+ix//8] |= 1 << (ix & 7)
                else:
                    if maxval < 56:
                        buf[offs:offs+8] = b'm\x00\x00\x00\x00\x00\x00\x00'
                    else:
                        buf[offs:offs+16] = b'M\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                    if cython.compiled:
                        # We'll use implicit casting in Cython
                        for ix in obj:
                            buf[offs+1+ix//8] |= 1 << (ix & 7)
                    else:
                        for x in obj:
                            x = int(x)
                            buf[offs+1+x//8] |= 1 << (x & 7)
                if maxval < 56:
                    offs += 8
                else:
                    offs += 16
                return offs
            else:
                # Else, same representation as a tuple of sorted items, only backed in-memory by a frozenset
                if isinstance(obj, npndarray):
                    tup = numpy.unique(obj)
                else:
                    tup = sorted(obj)
                return mapped_tuple.pack_into(tup, buf, offs, idmap, implicit_offs)
        else:
            # Same representation as a tuple of items, only backed in-memory by a frozenset, but
            # sorted by their stable hash value, in case these aren't all numeric objects.
            if all_float:
                if isinstance(obj, npndarray):
                    obj = numpy.unique(obj)
                else:
                    obj = sorted(obj)
            else:
                obj = sorted(obj, key=_stable_hash)
            return mapped_tuple.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    @cython.locals(
        i=int, j=int, offs=cython.longlong,
        pybuf='Py_buffer', pbuf='const unsigned char *', b=cython.uchar, fs_type=cython.uchar)
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
            if six.PY3 and not cython.compiled:
                fs_type = bytes([pbuf[offs]])
            else:
                fs_type = pbuf[offs]
            if fs_type == b'm' or fs_type == b'M':
                # inline bitmap
                if fs_type == b'm':
                    fs_size = 7
                elif fs_type == b'M':
                    fs_size = 15
                else:
                    raise ValueError("Unknown set type %r" % fs_type)
                if cython.compiled and offs+fs_size >= pybuf.len:
                    raise IndexError("Object spans beyond buffer end")
                rv = []
                for i in range(fs_size):
                    if six.PY3:
                        b = pbuf[offs+1+i]
                    else:
                        b = ord(pbuf[offs + 1 + i])
                    if b:
                        for j in range(8):
                            if b & (1<<j):
                                rv.append(i*8+j)
                return frozenset(rv)
            else:
                # unpack a list, build a set from it
                return mapped_list.unpack_from(buf, offs, idmap, klass=frozenset)
        finally:
            if cython.compiled:
                if type(buf) is buffer:
                    PyBuffer_Release(cython.address(pybuf))  # lint:ok

_struct_l_Q = cython.declare(object, struct.Struct('<Q'))
_struct_l_I = cython.declare(object, struct.Struct('<I'))

class mapped_tuple(tuple):
    cython.declare(
        __weakref__=object,
    )

    @classmethod
    @cython.locals(widmap = StrongIdMap, pindex = 'long[:]',
        rel_offs = cython.longlong, min_offs = cython.longlong, max_offs = cython.longlong,
        offs = cython.Py_ssize_t, implicit_offs = cython.Py_ssize_t, val_offs = cython.Py_ssize_t,
        i = cython.Py_ssize_t, iminval = cython.longlong, imaxval = cython.longlong)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0,
            array = array.array):
        baseoffs = offs
        objlen = len(obj)
        if isinstance(obj, npndarray):
            all_int = all_intlong = all_float = 0
            obj_dtype = obj.dtype
            if obj_dtype.isbuiltin:
                dtype = obj_dtype.char
                cdtype = cython.cast('const char*', dtype)[0]
                if cdtype in ('l', 'I', 'i', 'H', 'h', 'B', 'b'):
                    all_int = all_intlong = 1
                    all_float = 0
                elif cdtype == 'L':
                    all_int = all_float = 0
                    all_intlong = 1
                elif cdtype in ('d', 'f'):
                    all_int = all_intlong = 0
                    all_float = 1
                if all_int or all_intlong or all_float:
                    # translate l -> q
                    if cdtype == 'l':
                        buf[offs] = 'q'
                    elif cdtype == 'L':
                        buf[offs] = 'Q'
                    else:
                        buf[offs] = dtype
        else:
            all_int = all_intlong = all_float = 1
            minval = maxval = 0
            for x in obj:
                if all_int and type(x) is not int:
                    all_int = 0
                if all_intlong and type(x) is not int and type(x) is not long:
                    all_intlong = 0
                if all_float and type(x) is not float:
                    all_float = 0
                if not all_int and not all_intlong and not all_float:
                    break
                elif all_intlong:
                    if x < minval:
                        minval = x
                    if x > maxval:
                        maxval = x

            if all_int or all_intlong:
                try:
                    iminval = minval
                    imaxval = maxval
                    if 0 <= iminval and imaxval <= 0xFF:
                        # inline unsigned bytes
                        dtype = 'B'
                        buf[offs] = ord(dtype)
                    elif -0x80 <= iminval and imaxval <= 0x7F:
                        # inline signed bytes
                        dtype = 'b'
                        buf[offs] = ord(dtype)
                    elif 0 <= iminval and imaxval <= 0xFFFF:
                        # inline unsigned shorts
                        dtype = 'H'
                        buf[offs] = ord(dtype)
                    elif -0x8000 <= iminval and imaxval <= 0x7FFF:
                        # inline signed shorts
                        dtype = 'h'
                        buf[offs] = ord(dtype)
                    elif -0x80000000 <= iminval and imaxval <= 0x7FFFFFFF:
                        # inline signed ints
                        dtype = 'i'
                        buf[offs] = ord(dtype)
                    elif 0 <= iminval and imaxval <= cython.cast(cython.longlong, 0xFFFFFFFF):
                        # inline unsigned ints
                        dtype = 'I'
                        buf[offs] = ord(dtype)
                    elif (cython.cast(cython.longlong, -0x8000000000000000) <= iminval
                            and imaxval <= cython.cast(cython.longlong, 0x7FFFFFFFFFFFFFFF)):
                        # inline signed int64 list
                        buf[offs] = ord('q')
                        dtype = 'l'
                    else:
                        raise OverflowError
                except OverflowError:
                    if 0 <= minval and maxval <= 0xFFFFFFFFFFFFFFFF:
                        # inline unsigned int64 list
                        buf[offs] = ord('Q')
                        dtype = 'L'
                    elif all_int:
                        # inline sorted int64 list
                        buf[offs] = ord('q')
                        dtype = 'l'
                    else:
                        # longs are tricky, give up
                        all_int = all_intlong = 0
        if all_int or all_intlong:
            if dtype == 'l' or dtype == 'L':
                buf[offs+1:offs+8] = _struct_l_Q.pack(objlen)[:7]
                offs += 8
            elif objlen < 0xFFFFFF:
                buf[offs+1:offs+4] = _struct_l_I.pack(objlen)[:3]
                offs += 4
            else:
                buf[offs+1:offs+8] = '\xff\xff\xff\xff\xff\xff'
                buf[offs+8:offs+12] = _struct_l_Q.pack(objlen)
                offs += 12

            if isinstance(obj, npndarray):
                a = obj
            else:
                a = array(dtype, obj)
            abuf = buffer(a)
            if six.PY3:
                size_bytes = len(abuf) * abuf.itemsize
            else:
                size_bytes = len(abuf)
            buf[offs:offs+size_bytes] = abuf
            offs += size_bytes
            offs = (offs + 7) // 8 * 8
            return offs
        elif all_float:
            if isinstance(obj, npndarray):
                a = obj
                # dtype already set when inspecting obj's dtype
            else:
                a = array('d', obj)
                buf[offs] = ord('d')
            buf[offs+1:offs+8] = _struct_l_Q.pack(objlen)[:7]
            offs += 8
            abuf = buffer(a)
            buf[offs:offs+len(abuf)] = abuf
            offs += len(abuf)
            offs = (offs + 7) // 8 * 8
            return offs
        else:
            # inline object tuple
            use_narrow = False
            buf[offs] = ord('t')
            buf[offs+1:offs+8] = _struct_l_Q.pack(objlen)[:7]
            offs += 8

            # None will be represented with an offset of 1, which is an impossible offset
            # (it would point into this tuple's header, 0 would be the tuple itself so it's valid)
            indexoffs = offs
            pindex = index = npempty(len(obj), npint64)
            index_buffer = buffer(index)
            offs += len(index_buffer)

            if idmap is None:
                idmap = StrongIdMap()
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
            else:
                widmap = None

            # Get a sense of whether we can use narrow pointers
            min_offs = max_offs = 0
            for i, x in enumerate(obj):
                if x is None:
                    rel_offs = 1
                else:
                    # these are wrapped objects, not plain objects, so make sure they have distinct xid
                    xid = wrapped_id(x)
                    if xid not in idmap:
                        # Mark for later
                        pindex[i] = 1
                        continue
                    else:
                        val_offs = idmap[xid]
                    rel_offs = val_offs - (baseoffs + implicit_offs)
                pindex[i] = rel_offs
                if rel_offs < min_offs:
                    min_offs = rel_offs
                elif rel_offs > max_offs:
                    max_offs = rel_offs

            if min_offs >= -0x80000000 and max_offs <= 0x7fffffff and (len(buf) - baseoffs) <= 0x7fffffff:
                # We can use narrow pointers, guaranteed
                use_narrow = True
                offs = indexoffs + len(index_buffer) // 2
                offs += (8 - offs & 7) & 7
                dataoffs = offs
                buf[baseoffs] = ord('T')

            dataoffs = offs
            mapped_object_ = mapped_object
            pack_into = mapped_object_.pack_into
            for i,x in enumerate(obj):
                if x is None:
                    rel_offs = 1
                elif pindex[i] != 1:
                    # Already done, was a hit on the idmap
                    continue
                else:
                    # these are wrapped objects, not plain objects, so make sure they have distinct xid
                    xid = wrapped_id(x)
                    if xid not in idmap:
                        idmap[xid] = val_offs = offs + implicit_offs
                        if widmap is not None:
                            widmap.link(xid, x)
                        mx = mapped_object_(x)
                        offs = pack_into(mx, buf, offs, idmap, implicit_offs)
                    else:
                        val_offs = idmap[xid]
                    rel_offs = val_offs - (baseoffs + implicit_offs)
                pindex[i] = rel_offs
                if rel_offs < min_offs:
                    min_offs = rel_offs
                elif rel_offs > max_offs:
                    max_offs = rel_offs
            del pindex

            if not use_narrow and offs == dataoffs and min_offs >= -0x80000000 and max_offs <= 0x7fffffff:
                # it fits in 32-bits, and the index can shrink since we added no data, so shrink it
                use_narrow = True
                offs = indexoffs + len(index_buffer) // 2
                offs += (8 - offs & 7) & 7
                dataoffs = offs
                buf[baseoffs] = 'T'
            if use_narrow:
                del index_buffer
                index = index.astype(npint32)
                index_buffer = buffer(index)

            # write index
            buf[indexoffs:indexoffs+len(index_buffer)] = index_buffer

            return offs

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        if idmap is None:
            idmap = {}
        if offs in idmap:
            return idmap[offs]
        rv = idmap[offs] = mapped_list.unpack_from(buf, offs, idmap, klass=tuple)
        return rv

class mapped_list(list):
    cython.declare(
        __weakref__=object,
    )

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        # Same format as tuple, only different base type
        return mapped_tuple.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    @cython.locals(lrv = list, dchar = cython.char, objlen = cython.Py_ssize_t, absix = object,
        offs = cython.Py_ssize_t, baseoffs = cython.Py_ssize_t, ix = cython.Py_ssize_t)
    def unpack_from(cls, buf, offs, idmap = None, array = array.array, klass = list):
        if idmap is not None and offs in idmap:
            return idmap[offs]

        baseoffs = offs
        buf = _likerobuffer(buf)
        if six.PY3:
            dcode = chr(buf[offs])
            dcode_enc = dcode.encode()
            if cython.compiled:
                dchar = dcode_enc[0]
            else:
                dchar = dcode_enc
        else:
            dcode = buf[offs]
            dchar = cython.cast('const char*', dcode)[0]
        if six.PY3:
            buf = buf.tobytes() # FIXME this copies the data
        if dchar in (b'B', b'b'):
            itemsize = 1
        elif dchar in (b'H', b'h'):
            itemsize = 2
        elif dchar in (b'I', b'i', b'T'):
            itemsize = 4
        elif dchar in (b'Q', b'q', b'd', b't'):
            itemsize = 8
        else:
            raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
        if dchar in (b'B',b'b',b'H',b'h',b'I',b'i'):
            dtype = dcode
            objlen, = _struct_l_I.unpack(buf[offs:offs+4])
            objlen >>= 8
            offs += 4
            if objlen == 0xFFFFFF:
                objlen = _struct_l_Q.unpack_from(buf, offs)
                offs += 8
            rv = array(dtype, buf[offs:offs+itemsize*objlen])
        elif dchar == b'q' or dchar == b'Q':
            if dchar == b'q':
                dtype = b'l'
            elif dchar == b'Q':
                dtype = b'L'
            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
            objlen, = _struct_l_Q.unpack(buf[offs:offs+8])
            objlen >>= 8
            offs += 8
            rv = array(dtype, buf[offs:offs+itemsize*objlen])
        elif dchar == b'd':
            dtype = b'd'
            objlen, = _struct_l_Q.unpack(buf[offs:offs+8])
            objlen >>= 8
            offs += 8
            rv = array(dtype, buf[offs:offs+itemsize*objlen])
        elif dchar == b't' or dchar == b'T':
            if dchar == b't':
                dtype = b'l'
            elif dchar == b'T':
                dtype = b'i'
            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))

            objlen, = _struct_l_Q.unpack(buf[offs:offs+8])
            objlen >>= 8
            offs += 8

            index = array(dtype, buf[offs:offs+itemsize*objlen])

            if idmap is None:
                idmap = {}

            unpack_from = _mapped_object_unpack_from
            if klass is set or klass is frozenset:
                # Will turn recursive references into sets, but recursive references aren't
                # easy to build with frozensets, so we're cool
                idmap[baseoffs] = srv = set()
                for i,ix in enumerate(index):
                    if ix == 1:
                        srv.add(None)
                    else:
                        absix = ix + baseoffs
                        if absix in idmap:
                            obj = idmap[absix]
                        else:
                            obj = unpack_from(buf, absix, idmap)
                            idmap[absix] = obj
                        srv.add(obj)
                if klass is frozenset:
                    rv = frozenset(srv)
                else:
                    rv = srv
                return rv
            else:
                # Can't make perfect recursive references here, we'll do what we can
                idmap[baseoffs] = lrv = ([None] * objlen)
                for i,ix in enumerate(index):
                    if ix != 1:
                        absix = ix + baseoffs
                        if absix in idmap:
                            lrv[i] = idmap[absix]
                        else:
                            lrv[i] = idmap[absix] = unpack_from(buf, absix, idmap)
                if klass is tuple:
                    rv = tuple(lrv)
                elif klass is list:
                    rv = lrv
                else:
                    rv = klass(lrv)
                return rv
        else:
            raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
        if type(rv) is not klass:
            # Cython can specialize each conversion, so we do this ugly switch
            if klass is list:
                rv = list(rv)
            elif klass is tuple:
                rv = tuple(rv)
            elif klass is frozenset:
                rv = frozenset(rv)
            elif klass is set:
                rv = set(rv)
            else:
                rv = klass(rv)
        return rv

class mapped_dict(dict):
    cython.declare(
        __weakref__=object,
    )

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


_TUPLE_SEED = cython.declare(cython.ulonglong, 1626619511096549620)
_FSET_SEED  = cython.declare(cython.ulonglong, 8212431769940327799)

if not cython.compiled:
    globals()['isinf'] = math.isinf
    globals()['isnan'] = math.isnan

@cython.locals(hval=cython.ulonglong, trunc_key=cython.longlong,
    truncated=cython.bint, flt=cython.double, mant=cython.double)
def _stable_hash(key):
    """
    Compute a hash for object ``key`` in a way that is portable across processes and implementations.
    The computed hash should be appropriate for use in persistent data structures.

    Stable hashing is implemented for:

    * :class:`basestring`
    * :class:`int`
    * :class:`long`
    * :class:`float`
    * :class:`tuple` and :class:`proxied_tuple`
    * :class:`frozenset` and :class:`proxied_frozenset`

    Containers need to have hashable contents to be hashable.

    :rtype: int
    :returns: A 64-bit hash value
    """
    if key is None:
        hval = 1
    elif isinstance(key, basestring):
        hval = xxhash.xxh64(safe_utf8(key)).intdigest()
    elif isinstance(key, (int, long)):
        try:
            hval = key
        except OverflowError:
            hval = key & 0xFFFFFFFFFFFFFFFF
    elif isinstance(key, float):
        flt = key
        truncated = False
        try:
            trunc_key = int(flt)
            if trunc_key == flt:
                hval = trunc_key
                truncated = True
        except (OverflowError, ValueError):
            pass

        if not truncated:
            mant, expo = frexp(key)
            if expo < 0:
                # A double's exponent is usually limited to [-1024, 1024]
                expo += 0xFFFF
            if isinf(mant):
                mant = 1. if mant > 0 else -1.
            elif isnan(mant):
                mant = 2.
            hval = _mix_hash(expo, cython.cast(cython.longlong, mant * 0xFFFFFFFFFFFF))
    elif isinstance(key, (tuple, frozenset, proxied_tuple, proxied_frozenset)):
        if isinstance(key, (frozenset, proxied_frozenset)):
            hval = _FSET_SEED
        else:
            hval = _TUPLE_SEED

        for value in key:
            hval = _mix_hash(hval, _stable_hash(value))
    else:
        raise TypeError("unhashable type: %s" % type(key).__name__)

    if not cython.compiled:
        # Make sure it fits in a uint64
        hval = hval & 0xFFFFFFFFFFFFFFFF

    return hval if hval != 0 else 1


@cython.locals(idx=int)
def _enum_keys(obj):
    for idx, key in enumerate(iterkeys(obj)):
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

_DICT_HEADER_PACKER = cython.declare(object, struct.Struct('=Q'))
_DICT_HEADER_SIZE = cython.declare(cython.Py_ssize_t, _DICT_HEADER_PACKER.size)

if six.PY3:
    def cmp(a, b):
        return (a > b) - (a < b)

@cython.cclass
class proxied_dict(object):

    cython.declare(
        __weakref__=object,
        objmapper=object,
        vlist='proxied_list',
        buf=object,
        offs=cython.Py_ssize_t,
    )

    @property
    def buffer(self):
        return self.buf

    @property
    def offset(self):
        return self.offs

    def __init__(self, buf, offs, objmapper, vlist):
        self.objmapper = objmapper
        self.vlist = vlist
        self.buf = buf
        self.offs = offs

    @classmethod
    @cython.locals(widmap = StrongIdMap, cur_offs = cython.Py_ssize_t, implicit_offs = cython.Py_ssize_t)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        basepos = cur_offs = offs
        cur_offs += _DICT_HEADER_SIZE
        iobuf = BufferIO(buf, cur_offs)
        cur_offs += cython.cast(cython.Py_ssize_t, ObjectIdMapper.build(
            _enum_keys(obj), iobuf,
            return_mapper=False,
            idmap=idmap,
            implicit_offs=implicit_offs + cur_offs))
        _DICT_HEADER_PACKER.pack_into(buf, basepos, cur_offs - basepos)
        return proxied_list.pack_into([obj[k] for k in iterkeys(obj)], buf, cur_offs, idmap, implicit_offs)

    @classmethod
    @cython.locals(values_offs = cython.Py_ssize_t, ioffs = cython.Py_ssize_t)
    def unpack_from(cls, buf, offs, idmap = None):
        ioffs = offs
        values_offs, = _DICT_HEADER_PACKER.unpack_from(buf, offs)
        objmapper = ObjectIdMapper.map_buffer(buf, ioffs + _DICT_HEADER_SIZE)
        vlist = proxied_list.unpack_from(buf, ioffs + values_offs, idmap)
        return proxied_dict(buf, offs, objmapper, vlist)

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


_BUFFER_HEADER_PACKER = cython.declare(object, struct.Struct('=Q'))
_BUFFER_HEADER_SIZE = cython.declare(cython.Py_ssize_t, _BUFFER_HEADER_PACKER.size)


class proxied_buffer(object):
    """
    Shared-buffer implementation for byte buffers
    """

    @classmethod
    @cython.locals(cur_offs = cython.Py_ssize_t, objlen = cython.Py_ssize_t)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        """
        Packs buffer data from ``obj`` into ``buf``. Does not wrap it in :term:`RTTI`.

        See `mapped_object.pack_into` for argument details.
        """
        cur_offs = offs
        objlen = len(obj)
        _BUFFER_HEADER_PACKER.pack_into(buf, offs, objlen)
        cur_offs += _BUFFER_HEADER_SIZE

        end_offs = cur_offs + objlen
        buf[cur_offs:end_offs] = obj

        return end_offs

    @classmethod
    @cython.locals(cur_offs = cython.Py_ssize_t)
    def unpack_from(cls, buf, offs, idmap = None):
        """
        Unpacks buffer data from ``buf`` and returns a buffer slice of ``buf`` that contains the data.

        See `mapped_object.unpack_from` for argument details.
        """
        cur_offs = offs
        size, = _BUFFER_HEADER_PACKER.unpack_from(buf, offs)
        cur_offs += _BUFFER_HEADER_SIZE

        return buffer(buf, cur_offs, size)


_NDARRAY_HEADER_PACKER = cython.declare(object, struct.Struct('=QQ'))
_NDARRAY_HEADER_SIZE = cython.declare(cython.Py_ssize_t, _NDARRAY_HEADER_PACKER.size)

_NDARRAY_STANDARD_CODES_TO_DTYPE = cython.declare(tuple, (
    # Maintain the order, it's important
    numpy.uint64,
    numpy.int64,
    numpy.uint32,
    numpy.int32,
    numpy.uint16,
    numpy.int16,
    numpy.uint8,
    numpy.int8,
    numpy.float64,
    numpy.float32,
))
_NDARRAY_STANDARD_DTYPES_TO_CODE = cython.declare(dict, dict([
    (numpy.dtype(dt).str, i)
    for i, dt in enumerate(_NDARRAY_STANDARD_CODES_TO_DTYPE)
]))
NDARRAY_STANDARD_CODES_TO_DTYPE = _NDARRAY_STANDARD_CODES_TO_DTYPE
NDARRAY_STANDARD_DTYPES_TO_CODE = _NDARRAY_STANDARD_DTYPES_TO_CODE


class proxied_ndarray(object):

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
    @cython.locals(cur_offs = cython.Py_ssize_t, implicit_offs = cython.Py_ssize_t, baseoffs = cython.Py_ssize_t)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        cur_offs = baseoffs = header_offs = offs
        cur_offs += _NDARRAY_HEADER_SIZE

        shape = obj.shape
        if len(shape) != 1:
            cur_offs = mapped_tuple.pack_into(shape, buf, cur_offs)
        dtype_offs = cur_offs - baseoffs

        dtype_params = cls._make_dtype_params(obj.dtype)
        if isinstance(dtype_params, basestring):
            dtype_params = _NDARRAY_STANDARD_DTYPES_TO_CODE.get(dtype_params, dtype_params)
        cur_offs = mapped_object.pack_into(dtype_params, buf, cur_offs)
        data_offs = cur_offs - baseoffs

        _NDARRAY_HEADER_PACKER.pack_into(buf, header_offs, dtype_offs, data_offs)
        return proxied_buffer.pack_into(buffer(obj), buf, cur_offs)


    @classmethod
    @cython.locals(baseoffs = cython.Py_ssize_t, dtype_offs = cython.Py_ssize_t, data_offs = cython.Py_ssize_t,
        shape_offs = cython.Py_ssize_t)
    def unpack_from(cls, buf, offs, idmap = None):
        baseoffs = offs
        dtype_offs, data_offs = _NDARRAY_HEADER_PACKER.unpack_from(buf, offs)

        shape_offs = _NDARRAY_HEADER_SIZE
        if dtype_offs != shape_offs:
            shape = mapped_tuple.unpack_from(buf, baseoffs + shape_offs)
        else:
            shape = None
        dtype_params = _mapped_object_unpack_from(buf, baseoffs + dtype_offs)

        data = proxied_buffer.unpack_from(buf, baseoffs + data_offs)

        if type(dtype_params) is int:
            dtype = _NDARRAY_STANDARD_CODES_TO_DTYPE[cython.cast(int, dtype_params)]
        else:
            dtype = numpy.dtype(dtype_params)

        ndarray = npfrombuffer(data, dtype)
        if shape is not None:
            ndarray = ndarray.reshape(shape)
        return ndarray

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

    for i in range(min(alen, blen)):
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
        __weakref__ = object,
        buf = object,
        pybuf = 'Py_buffer',
        offs = cython.Py_ssize_t,
        elem_start = cython.longlong,
        elem_end = cython.longlong,
        elem_step = cython.longlong
    )

    @property
    def buffer(self):
        return self.buf

    @property
    def offset(self):
        return self.offs

    def __dealloc__(self):
        if cython.compiled:
            if self.pybuf.buf != cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok
                self.pybuf.buf = cython.NULL

    @cython.ccall
    @cython.locals(dataoffs = cython.Py_ssize_t, dcode = cython.char, pbuf = 'const char *',
        itemsize = cython.uchar, objlen = cython.Py_ssize_t)
    def _metadata(self,
        itemsizes = dict([(dtype, array.array(dtype.decode(), []).itemsize) for dtype in (b'B',b'b',b'H',b'h',b'I',b'i',b'l',b'L',b'd')])):

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

            elif dcode in ('q', 'Q', 'd', 't', 'T'):
                objlen = cython.cast(cython.p_longlong, pbuf + dataoffs)[0] >> 8
                dataoffs += 8
                if dcode == 'T':
                    itemsize = 4
                else:
                    itemsize = 8
                return dcode, objlen, itemsize, dataoffs, None

            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))

        else:
            # Python version
            dataoffs = self.offs
            buf = self.buf

            if six.PY3 and not cython.compiled:
                dcode = bytes([buf[dataoffs]])
            else:
                dcode = buf[dataoffs]

            if dcode in (b'B',b'b',b'H',b'h',b'I',b'i'):
                objlen, = struct.unpack('<I', bytes(buf[dataoffs+1:dataoffs+4]) + b'\x00')
                dataoffs += 4
                if objlen == 0xFFFFFF:
                    objlen = struct.unpack_from('<Q', buf, dataoffs)
                    dataoffs += 8
                return dcode, objlen, itemsizes[dcode], dataoffs, struct.Struct(dcode)

            elif dcode == b'q':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['l'], dataoffs, struct.Struct('q')

            elif dcode == b'Q':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['L'], dataoffs, struct.Struct('Q')

            elif dcode == b'd':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['d'], dataoffs, struct.Struct('d')

            elif dcode == b't':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['l'], dataoffs, struct.Struct('l')

            elif dcode == b'T':
                objlen, = struct.unpack('<Q', buf[dataoffs+1:dataoffs+8] + '\x00')
                dataoffs += 8
                return dcode, objlen, itemsizes['i'], dataoffs, struct.Struct('i')

            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))

    def __cinit__(self, buf, offs, idmap = None, elem_start = 0, elem_end = 0, elem_step = 0):
        if cython.compiled:
            self.pybuf.buf = cython.NULL

    @cython.locals(offs = cython.Py_ssize_t, idmap = dict, elem_start = cython.longlong,
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
            if self.pybuf.buf != cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok
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

    @cython.locals(fast = cython.bint, dcode = cython.char, index = cython.longlong,
        objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def getter(self, proxy_into = None, fast = False):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()

        if proxy_into is None and fast and dcode in ('t', 'T'):
            proxy_into = _GenericProxy_new(_GenericProxy, None, 0, 0)
            if not cython.compiled:
                proxy_into.buf = None

        @cython.locals(index = cython.longlong)
        def getter(index):
            return self._c_getitem(index, dcode, objlen, itemsize, dataoffs, _struct, proxy_into)

        return getter

    @cython.ccall
    @cython.locals(dcode = cython.char, index = cython.longlong,
        objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def _getitem(self, index):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        return self._c_getitem(index, dcode, objlen, itemsize, dataoffs, _struct, None)

    @cython.inline
    @cython.cfunc
    @cython.locals(obj_offs = cython.Py_ssize_t, dcode = cython.char, index = cython.longlong,
        objlen = cython.longlong, xlen = cython.longlong, step = cython.longlong,
        lpindex = "const long *",
        ipindex = "const int *",
        dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def _c_getitem(self, index, dcode, objlen, itemsize, dataoffs, _struct, proxy_into):
        xlen = objlen
        orig_index = index

        if self.elem_step != 0:
            if self.elem_end == self.elem_start:
                raise IndexError(orig_index)
            step = abs(self.elem_step)
            if self.elem_step > 0:
                xlen = (self.elem_end - self.elem_start - 1) // step + 1
            else:
                xlen = (self.elem_start - self.elem_end - 1) // step + 1

            index = self.elem_start + index * self.elem_step

            if ((self.elem_step < 0 and (index > self.elem_start or index <= self.elem_end))
                    or (self.elem_step > 0 and (index >= self.elem_end or index < self.elem_start))):
                raise IndexError(orig_index)

        if index < 0:
            index += xlen

        if index >= objlen or index < 0:
            raise IndexError(orig_index)

        if dcode in ('t', 'T'):
            if cython.compiled:
                if dcode == 't':
                    lpindex = cython.cast('const long *', cython.cast(cython.p_uchar, self.pybuf.buf) + dataoffs)
                    if lpindex[index] == 1:
                        return None
                    obj_offs = self.offs + lpindex[index]
                elif dcode == 'T':
                    ipindex = cython.cast('const int *', cython.cast(cython.p_uchar, self.pybuf.buf) + dataoffs)
                    if ipindex[index] == 1:
                        return None
                    obj_offs = self.offs + ipindex[index]
            else:
                index_offs = dataoffs + itemsize * int(index)
                rel_offs = _struct.unpack_from(self.buf, index_offs)[0]
                if rel_offs == 1:
                    return None
                obj_offs = self.offs + rel_offs
        else:
            obj_offs = dataoffs + itemsize * cython.cast(cython.size_t, int(index))

        if dcode in ('t', 'T'):
            res = _mapped_object_unpack_from(self.buf, obj_offs, None, proxy_into)
        elif cython.compiled:
            if dcode == 'B':
                res = cython.cast(cython.p_uchar,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'b':
                res = cython.cast(cython.p_schar,
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
            elif dcode == 'Q':
                res = cython.cast(cython.p_ulonglong,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            elif dcode == 'd':
                res = cython.cast(cython.p_double,
                    cython.cast(cython.p_uchar, self.pybuf.buf) + obj_offs)[0]  # lint:ok
            else:
                raise ValueError("Inconsistent data, unknown type code %r" % (dcode,))
        else:
            res = _struct.unpack_from(self.buf, obj_offs)[0]

        return res

    @cython.ccall
    def _make_empty(self):
        return []

    @cython.locals(start = cython.Py_ssize_t, end = cython.Py_ssize_t, step = cython.Py_ssize_t)
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

            return type(self)(
                self.buf,
                self.offs,
                None,
                start, end, step)

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
            return (self.elem_start - self.elem_end - 1) // (-self.elem_step) + 1
        else:
            return (self.elem_end - self.elem_start - 1) // self.elem_step + 1

    def __nonzero__(self):
        return len(self) > 0

    def __bool__(self):
        return len(self) > 0

    def __setitem__(self, index, value):
        raise TypeError("Proxy objects are read-only")

    def __delitem__(self, index):
        raise TypeError("Proxy objects are read-only")

    @cython.locals(i=cython.longlong,
        dcode = cython.char, objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def __iter__(self):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        for i in range(len(self)):
            yield self._c_getitem(i, dcode, objlen, itemsize, dataoffs, _struct, None)

    @cython.locals(i=cython.longlong,
        dcode = cython.char, objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar,
        pmask = 'const unsigned char[:]')
    def iter(self, proxy_into=None, mask=None):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        if mask is not None:
            pmask = mask
        for i in range(len(self)):
            if mask is not None and not pmask[i]:
                continue
            yield self._c_getitem(i, dcode, objlen, itemsize, dataoffs, _struct, proxy_into)

    @cython.locals(i=cython.longlong,
        dcode = cython.char, objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar,
        pmask = 'const unsigned char[:]')
    def iter_fast(self, mask=None):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        if dcode in ('t', 'T'):
            proxy_into = _GenericProxy_new(_GenericProxy, None, 0, 0)
            if not cython.compiled:
                proxy_into.buf = None
        else:
            proxy_into = None
        if mask is not None:
            pmask = mask
        for i in range(len(self)):
            if mask is not None and not pmask[i]:
                continue
            yield self._c_getitem(i, dcode, objlen, itemsize, dataoffs, _struct, proxy_into)

    @cython.locals(l=cython.Py_ssize_t,
        dcode = cython.char, objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def __reversed__(self):
        l = len(self)
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        if l > 0:
            for i in range(l - 1, -1, -1):
                yield self._c_getitem(i, dcode, objlen, itemsize, dataoffs, _struct, None)

    @cython.locals(i=cython.longlong,
        dcode = cython.char, objlen = cython.longlong, dataoffs = cython.Py_ssize_t, itemsize = cython.uchar)
    def __contains__(self, item):
        dcode, objlen, itemsize, dataoffs, _struct = self._metadata()
        if dcode in ('t', 'T'):
            proxy_into = _GenericProxy_new(_GenericProxy, None, 0, 0)
            if not cython.compiled:
                proxy_into.buf = None
        else:
            proxy_into = None
        for i in range(len(self)):
            if self._c_getitem(i, dcode, objlen, itemsize, dataoffs, _struct, proxy_into) == item:
                return True
        return False

    def __repr__(self):
        return "proxied_list(%s)" % self

    def __str__(self):
        return "[%s]" % ",".join([str(x) for x in self])

if not cython.compiled:
    setattr(proxied_list, '__eq__', getattr(proxied_list, '_eq'))
    setattr(proxied_list, '__ne__', getattr(proxied_list, '_ne'))

if not cython.compiled:
    def _popcountll(x):
        i = 0
        while x != 0:
            if (x & 1) != 0:
                i += 1
            x >>= 1
        return i

    globals()['popcount'] = _popcountll

@cython.cclass
class proxied_frozenset(object):

    cython.declare(objlist='proxied_list',
        bitrep_lo=cython.ulonglong, bitrep_hi=cython.ulonglong,
        bitlen=cython.Py_ssize_t)

    def __init__(self, objlist, bitrep_lo=0, bitrep_hi=0):
        self.objlist = objlist
        self.bitrep_lo = bitrep_lo
        self.bitrep_hi = bitrep_hi

        if self.objlist is None:
            self.bitlen = popcount(self.bitrep_lo) + popcount(self.bitrep_hi)

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        return mapped_frozenset.pack_into(obj, buf, offs, idmap, implicit_offs)

    @classmethod
    @cython.locals(
        i=int, j=int, offs=cython.longlong,
        pybuf='Py_buffer', pbuf='const unsigned char *', b=cython.uchar,
        bitrep_lo=cython.ulonglong, bitrep_hi=cython.ulonglong)
    def unpack_from(cls, buf, offs, idmap = None):
        buf = _likerobuffer(buf)
        try:
            if cython.compiled:
                pybuf.buf = cython.NULL
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                pbuf = cython.cast(cython.p_uchar, pybuf.buf)
                if offs >= pybuf.len:
                    raise IndexError("Offset out of range")
            else:
                pbuf = buf

            if pbuf[offs] == 'm':
                # inline bitmap (64 bits)
                if cython.compiled and offs+7 >= pybuf.len:
                    raise IndexError("Object spans beyond buffer end")
                bitrep_lo = cython.cast(cython.p_ulonglong, pbuf + offs)[0] >> 8
                return proxied_frozenset(None, bitrep_lo, 0)
            elif pbuf[offs] == 'M':
                # inline bitmap (128 bits)
                if cython.compiled and offs+15 >= pybuf.len:
                    raise IndexError("Object spans beyond buffer end")
                bitrep_lo = cython.cast(cython.p_ulonglong, pbuf + offs)[0] >> 8
                bitrep_hi = cython.cast(cython.p_ulonglong, pbuf + offs)[1]
                bitrep_lo |= (bitrep_hi & 0xff) << 56
                bitrep_hi >>= 8
                return proxied_frozenset(None, bitrep_lo, bitrep_hi)
            else:
                return proxied_frozenset(proxied_list.unpack_from(buf, offs, idmap))
        finally:
            if cython.compiled:
                if type(buf) is buffer and pybuf.buf != cython.NULL:
                    PyBuffer_Release(cython.address(pybuf))

    def copy(self):
        return self

    def __len__(self):
        if self.objlist is not None:
            return len(self.objlist)
        else:
            return self.bitlen

    @cython.cfunc
    @cython.locals(dcode=cython.char, offset=cython.size_t,
        start=cython.size_t, xlen=cython.size_t, hint=cython.size_t, equal=cython.bint)
    def _search_key(self, elem, dcode, offset, start, xlen, hint, equal=False):
        pindex = cython.cast(cython.p_char, self.objlist.pybuf.buf) + offset
        if dcode == 'Q':
            return _c_search_hkey_ui64(elem, pindex + start * 8, 8, xlen, hint, equal)
        elif dcode == 'q':
            return _c_search_hkey_i64(elem, pindex + start * 8, 8, xlen, hint, equal)
        elif dcode == 'I':
            return _c_search_hkey_ui32(elem, pindex + start * 4, 4, xlen, hint, equal)
        elif dcode == 'i':
            return _c_search_hkey_i32(elem, pindex + start * 4, 4, xlen, hint, equal)
        elif dcode == 'H':
            return _c_search_hkey_ui16(elem, pindex + start * 2, 2, xlen, hint, equal)
        elif dcode == 'h':
            return _c_search_hkey_i16(elem, pindex + start * 2, 2, xlen, hint, equal)
        elif dcode == 'B':
            return _c_search_hkey_ui8(elem, pindex + start, 1, xlen, hint, equal)
        elif dcode == 'b':
            return _c_search_hkey_i8(elem, pindex + start, 1, xlen, hint, equal)
        elif dcode == 'd':
            return _c_search_hkey_f64(elem, pindex + start * 8, 8, xlen, hint, equal)
        else:
            raise NotImplementedError("Unsupported data type for fast lookup: %s" % chr(dcode))

    @cython.cfunc
    @cython.locals(eint=cython.ulonglong)
    def _contains_compressed(self, elem):
        try:
            eint = int(elem)
        except (ValueError, TypeError, OverflowError):
            return False

        if eint != elem or eint >= 128 or eint < 0:
            return False
        return (eint < 64 and (self.bitrep_lo & (1 << eint)) != 0) or (
            (eint < 128 and (self.bitrep_hi & (1 << (eint - 64))) != 0))

    @cython.locals(lo=cython.Py_ssize_t, hi=cython.Py_ssize_t, mid=cython.Py_ssize_t,
        dcode=cython.char, objlen=cython.longlong, itemsize=cython.uchar,
        offset=cython.Py_ssize_t, h1=cython.ulonglong, h2=cython.ulonglong)
    def __contains__(self, elem):
        if self.objlist is None:
            return self._contains_compressed(elem)

        lo = 0
        hi = len(self.objlist)
        dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()

        if cython.compiled:
            if dcode not in ('t', 'T'):
                return self._search_key(elem, dcode, offset, lo, hi, hi // 2, True) < hi

        h2 = _stable_hash(elem)
        while lo < hi:
            mid = lo + ((hi - lo) >> 1)
            val = self.objlist._c_getitem(mid, dcode, objlen, itemsize, offset, _struct, None)
            h1 = _stable_hash(val)
            if h1 == h2:
                # Equal hashes - Look in both directions for a match, without binary search
                if val == elem:
                    return True
                while mid > 0:
                    val = self.objlist._c_getitem(mid, dcode, objlen, itemsize, offset, _struct, None)
                    h1 = _stable_hash(val)
                    if h1 != h2:
                        break
                    elif val == elem:
                        return True
                    mid -= 1

                mid = lo + ((hi - lo) >> 1)
                while mid < hi:
                    val = self.objlist._c_getitem(mid, dcode, objlen, itemsize, offset, _struct, None)
                    h1 = _stable_hash(val)
                    if h1 != h2:
                        break
                    elif val == elem:
                        return True
                    mid += 1
                return False
            elif h1 < h2:
                lo = mid + 1
            else:
                hi = mid
        return False

    @cython.locals(i=cython.Py_ssize_t, dcode=cython.char, objlen=cython.longlong,
        itemsize=cython.uchar, offset=cython.Py_ssize_t)
    def __iter__(self):
        if self.objlist is not None:
            dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
            for i in range(len(self)):
                yield self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None)
        else:
            if self.bitrep_lo:
                for i in range(64):
                    if self.bitrep_lo & (cython.cast(cython.ulonglong, 1) << i):
                        yield i
                    elif not self.bitrep_lo >> i:
                        break

            if self.bitrep_hi:
                for i in range(64):
                    if self.bitrep_hi & (cython.cast(cython.ulonglong, 1) << i):
                        yield i + 64
                    elif not self.bitrep_hi >> i:
                        break

    @cython.locals(i=cython.Py_ssize_t, out=set)
    @cython.returns(set)
    @cython.ccall
    def _add_to(self, out):
        if self.objlist is None:
            for i in range(64):
                if self.bitrep_lo & (cython.cast(cython.ulonglong, 1) << i):
                    out.add(i)
                elif not self.bitrep_lo >> i:
                    break
            for i in range(64):
                if self.bitrep_hi & (cython.cast(cython.ulonglong, 1) << i):
                    out.add(i + 64)
                elif not self.bitrep_hi >> i:
                    break
            return out

        dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
        for i in range(len(self)):
            out.add(self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None))
        return out

    @cython.ccall
    @cython.locals(pfset='proxied_frozenset')
    def _union_2(self, seq):
        if not seq:
            return self
        elif self.objlist is None and type(seq) is proxied_frozenset:
            pfset = cython.cast(proxied_frozenset, seq)
            if pfset.objlist is None:
                return proxied_frozenset(
                    None, self.bitrep_lo | pfset.bitrep_lo, self.bitrep_hi | pfset.bitrep_hi)

        return frozenset(self._add_to(set(seq)))

    @cython.ccall
    @cython.locals(pfset='proxied_frozenset', rv=set, set_seq=set, fs_seq=frozenset)
    def _intersect_2(self, seq):
        if self.objlist is None and type(seq) is proxied_frozenset:
            pfset = cython.cast(proxied_frozenset, seq)
            if pfset.objlist is None:
                return proxied_frozenset(
                    None, self.bitrep_lo & pfset.bitrep_lo, self.bitrep_hi & pfset.bitrep_hi)
        elif not seq:
            return proxied_frozenset(None, 0, 0)

        if len(self) > len(seq):
            rv = set(seq)
            if type(seq) is set:
                set_seq = seq
                for val in set_seq:
                    if val not in self:
                        rv.discard(val)
            elif type(seq) is frozenset:
                fs_seq = seq
                for val in fs_seq:
                    if val not in self:
                        rv.discard(val)
            else:
                for val in seq:
                    if val not in self:
                        rv.discard(val)
        else:
            rv = self._add_to(set())
            rv.intersection_update(seq)
        return frozenset(rv)

    @cython.ccall
    @cython.locals(pfset='proxied_frozenset')
    def _diff_2(self, seq):
        if not seq:
            return self
        elif self.objlist is None and type(seq) is proxied_frozenset:
            pfset = cython.cast(proxied_frozenset, seq)
            if pfset.objlist is None:
                return proxied_frozenset(
                    None, self.bitrep_lo & ~pfset.bitrep_lo, self.bitrep_hi & ~pfset.bitrep_hi)

        rv = self._add_to(set())
        rv.difference_update(seq)
        return frozenset(rv)

    @cython.ccall
    @cython.locals(pfset='proxied_frozenset')
    def _symdiff_2(self, seq):
        if not seq:
            return self
        elif self.objlist is None and type(seq) is proxied_frozenset:
            pfset = cython.cast(proxied_frozenset, seq)
            if pfset.objlist is None:
                return proxied_frozenset(
                    None, self.bitrep_lo ^ pfset.bitrep_lo, self.bitrep_hi ^ pfset.bitrep_hi)

        rv = self._add_to(set())
        rv.symmetric_difference_update(seq)
        return frozenset(rv)

    def union(self, *seqs):
        if not seqs:
            return self
        elif len(seqs) == 1:
            return self._union_2(seqs[0])
        else:
            return frozenset(self._add_to(set())).union(*seqs)

    def intersection(self, *seqs):
        if not seqs:
            return self
        elif len(seqs) == 1:
            return self._intersect_2(seqs[0])
        else:
            return frozenset(self._add_to(set())).intersection(*seqs)

    def difference(self, *seqs):
        if not seqs:
            return self
        elif len(seqs) == 1:
            return self._diff_2(seqs[0])
        else:
            return frozenset(self._add_to(set())).difference(*seqs)

    def symmetric_difference(self, *seqs):
        if not seqs:
            return self
        elif len(seqs) == 1:
            return self._symdiff_2(seqs[0])
        else:
            return frozenset(self._add_to(set())).symmetric_difference(*seqs)

    @cython.ccall
    @cython.locals(i=cython.Py_ssize_t, xlen=cython.Py_ssize_t,
        pfset='proxied_frozenset', dcode=cython.char,
        objlen=cython.longlong, itemsize=cython.uchar, offset=cython.Py_ssize_t)
    def _frozenset_eq(self, x):
        if isinstance(x, proxied_frozenset):
            pfset = cython.cast(proxied_frozenset, x)
            return (self.bitrep_lo == pfset.bitrep_lo and
                self.bitrep_hi == pfset.bitrep_hi and
                self.objlist == pfset.objlist)

        xlen = len(self)
        if not isinstance(x, (set, frozenset)) or xlen != len(x):
            return False
        elif self.objlist is None:
            if self.bitrep_lo:
                for i in range(64):
                    if not self.bitrep_lo >> i:
                        break
                    elif (self.bitrep_lo & (cython.cast(cython.ulonglong, 1) << i)) and i not in x:
                        return False

            if self.bitrep_hi:
                for i in range(64):
                    if not self.bitrep_hi >> i:
                        break
                    elif (self.bitrep_hi & (cython.cast(cython.ulonglong, 1) << i)) and (i + 64) not in x:
                        return False

            return True

        dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
        for i in range(xlen):
            if self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None) not in x:
                return False
        return True

    def __eq__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._frozenset_eq(seq)
        else:
            return cython.cast(proxied_frozenset, seq)._frozenset_eq(self)

    def __ne__(self, seq):
        return not (self == seq)

    @cython.ccall
    @cython.locals(strict_subset=cython.bint, i=cython.Py_ssize_t,
        j=cython.Py_ssize_t, tmp_idx=cython.Py_ssize_t,
        xlen=cython.Py_ssize_t, pfset='proxied_frozenset',
        seqlen=cython.Py_ssize_t, dcode=cython.char, objlen=cython.longlong,
        itemsize=cython.uchar, offset=cython.Py_ssize_t, dcode2=cython.char,
        oblen2=cython.longlong, itemsize2=cython.uchar, offset2=cython.Py_ssize_t,
        h1=cython.ulonglong, h2=cython.ulonglong)
    def _subset(self, seq, strict_subset):
        i = 0
        xlen = len(self)

        if isinstance(seq, proxied_frozenset):
            pfset = cython.cast(proxied_frozenset, seq)
            if pfset.objlist is None:
                if self.objlist is None:
                    # Both proxies use compressed representation.
                    return (self.bitrep_lo & pfset.bitrep_lo) == self.bitrep_lo and (
                        self.bitrep_hi & pfset.bitrep_hi) == self.bitrep_hi and (
                        not strict_subset or self.bitlen < pfset.bitlen)
                else:
                    dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
                    if dcode in ('t', 'T'):
                        # non-numeric typecode
                        return False

                    for i in range(xlen):
                        val = self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None)
                        if not pfset._contains_compressed(val):
                            return False
                    return not strict_subset or xlen < pfset.bitlen

            elif self.objlist is None:
                if self.bitrep_lo:
                    for i in range(64):
                        if not self.bitrep_lo >> i:
                            break
                        elif (self.bitrep_lo & (cython.cast(cython.ulonglong, 1) << i)) and i not in pfset:
                            return False

                if self.bitrep_hi:
                    for i in range(64):
                        if not self.bitrep_hi >> i:
                            break
                        elif (self.bitrep_hi & (cython.cast(cython.ulonglong, 1) << i)) and (i + 64) not in pfset:
                            return False

                return not strict_subset or xlen < len(pfset)

            # Both proxies use sorted lists.
            j = 0
            seqlen = len(pfset.objlist)
            if xlen > seqlen:
                return False

            dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
            dcode2, objlen2, itemsize2, offset2, _struct2 = pfset.objlist._metadata()

            if dcode2 not in ('t', 'T') and cython.compiled:
                # fast path, use the search_hkey variants
                for i in range(xlen):
                    val = self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None)
                    tmp_idx = pfset._search_key(val, dcode, offset, j, seqlen - j, j, True)
                    if tmp_idx >= seqlen - j:
                        return False
                    j = tmp_idx
            else:
                for i in range(xlen):
                    val = self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None)
                    h1 = _stable_hash(val)
                    while True:
                        val2 = pfset.objlist._c_getitem(j, dcode2, objlen2, itemsize2, offset2, _struct2, None)
                        h2 = _stable_hash(val2)
                        j += 1

                        if h1 == h2 and val == val2:
                            break
                        elif h1 < h2 or j >= seqlen:
                            return False
                        else:
                            # pfset[j] < val, skip as much as we can
                            while j < seqlen:
                                val = pfset.objlist._c_getitem(j, dcode2, objlen2, itemsize2, offset2, _struct2, None)
                                if _stable_hash(val) >= h1:
                                    break
                                j += 1
                            if j == seqlen:
                                return False
            return not strict_subset or xlen < seqlen

        # seq is not a proxied_frozenset
        try:
            seqlen = len(seq)
        except (TypeError, AttributeError):
            # seq is not a sequence, therefore this operation is meaningless.
            return False
        else:
            if self.objlist is None:
                if self.bitrep_lo:
                    for i in range(64):
                        if not self.bitrep_lo >> i:
                            break
                        elif (self.bitrep_lo & (cython.cast(cython.ulonglong, 1) << i)) and i not in seq:
                            return False

                if self.bitrep_hi:
                    for i in range(64):
                        if not self.bitrep_hi >> i:
                            break
                        elif (self.bitrep_hi & (cython.cast(cython.ulonglong, 1) << i)) and (i + 64) not in seq:
                            return False
            else:
                dcode, objlen, itemsize, offset, _struct = self.objlist._metadata()
                for i in range(xlen):
                    val = self.objlist._c_getitem(i, dcode, objlen, itemsize, offset, _struct, None)
                    if val not in seq:
                        return False

            return not strict_subset or xlen < seqlen

    def __lt__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._subset(seq, True)
        else:
            return cython.cast(proxied_frozenset, seq)._subset(self, True)

    def __le__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._subset(seq, False)
        else:
            return cython.cast(proxied_frozenset, seq)._subset(self, False)

    def __gt__(self, seq):
        return seq < self

    def __ge__(self, seq):
        return seq <= self

    def __nonzero__(self):
        return len(self) > 0

    def __or__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._union_2(seq)
        else:
            return cython.cast(proxied_frozenset, seq)._union_2(self)

    def __and__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._intersect_2(seq)
        else:
            return cython.cast(proxied_frozenset, seq)._intersect_2(self)

    def __sub__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._diff_2(seq)
        else:
            return cython.cast(proxied_frozenset, seq)._diff_2(self)

    def __xor__(self, seq):
        if isinstance(self, proxied_frozenset):
            return cython.cast(proxied_frozenset, self)._symdiff_2(seq)
        else:
            return cython.cast(proxied_frozenset, seq)._symdiff_2(self)

    def __repr__(self):
        return "proxied_frozenset([%s])" % ",".join(str(x) for x in self)

    __str__ = __repr__

if six.PY3:
    _cpython = sys.implementation.name == 'cpython'
else:
    _cpython = sys.subversion[0] == 'CPython'
is_cpython = cython.declare(cython.bint, _cpython)

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
                for i in range(len_):
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
        rv = buffer_with_offset(buf, dataoffs, objlen)
        if compressed:
            rv = lz4_decompress(rv)
        else:
            rv = bytes(rv)

    if idmap is not None:
        idmap[offs] = rv
    return rv

class mapped_bytes(bytes):
    """
    Shared-buffer implementation for byte strings.
    """

    @classmethod
    @cython.locals(
        offs = cython.longlong, implicit_offs = cython.longlong,
        objlen = cython.size_t, objcomplen = cython.size_t, obj = bytes, objcomp = bytes,
        pbuf = 'char *', pybuf='Py_buffer', compressed = cython.ushort,
        widmap = StrongIdMap)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        """
        Packs a byte string into ``buf``. Does not wrap it in :term:`RTTI`.

        See `mapped_object.pack_into` for argument details.
        """
        if idmap is not None:
            objid = shared_id(obj)
            idmap[objid] = offs + implicit_offs
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
                widmap.link(objid, obj)
        objlen = len(obj)

        if objlen > MIN_COMPRESS_THRESHOLD:
            objcomp = lz4_compress(obj)
            objcomplen = len(objcomp)
            if objcomplen < (objlen - objlen//3):
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
        """
        Unpacks a byte string from ``buf``.

        See `mapped_object.unpack_from`.
        """
        return _unpack_bytes_from_pybuffer(buf, offs, idmap)
_mapped_bytes = cython.declare(object, mapped_bytes)

class mapped_unicode(six.text_type):
    """
    Shared-buffer implementation for unicode strings
    """

    @classmethod
    @cython.locals(widmap = StrongIdMap)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        """
        Packs a unicode string into ``buf``. Does not wrap it in :term:`RTTI`.

        See `mapped_object.pack_into` for argument details.
        """
        if idmap is not None:
            objid = shared_id(obj)
            idmap[objid] = offs + implicit_offs
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
                widmap.link(objid, obj)

        return mapped_bytes.pack_into(obj.encode("utf8"), buf, offs, None, implicit_offs)

    @classmethod
    def unpack_from(cls, buf, offs, idmap = None):
        """
        Unpacks a unicode string from ``buf``.

        See `mapped_object.unpack_from` for argument details.
        """
        if idmap is not None and offs in idmap:
            return idmap[offs]

        rv = mapped_bytes.unpack_from(buf, offs).decode("utf8")
        if idmap is not None:
            idmap[offs] = rv
        return rv

class mapped_decimal(Decimal):
    PACKER = struct.Struct('=q')

    @classmethod
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, exponent = cython.longlong,
        sign = cython.uchar, widmap = StrongIdMap)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = shared_id(obj)
            idmap[objid] = offs + implicit_offs
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
                widmap.link(objid, obj)

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
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, timestamp = cython.longlong,
        widmap = StrongIdMap)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = shared_id(obj)
            idmap[objid] = offs + implicit_offs
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
                widmap.link(objid, obj)

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
    @cython.locals(offs = cython.longlong, implicit_offs = cython.longlong, timestamp = cython.longlong,
        widmap = StrongIdMap)
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        if idmap is not None:
            objid = shared_id(obj)
            idmap[objid] = offs + implicit_offs
            if isinstance(idmap, StrongIdMap):
                widmap = idmap
                widmap.link(objid, obj)

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
        proxied_frozenset: 'z',

        int : 'q',
        long : 'q',
        float : 'd',
        bytes : 's',
        six.text_type : 'u',
        datetime : 'v',
        date : 'V',
        Decimal : 'F',
        cDecimal : 'F',
        numpy.ndarray : 'n',
        buffer : 'r',

        dict : 'm',
        collections.defaultdict : 'm',
        set : 'Z',
    }

    def p(s):
        return s, (s.size + 7) // 8 * 8 - s.size

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
        'M' : (proxied_dict.pack_into, proxied_dict.unpack_from, proxied_dict),
        'z' : (proxied_frozenset.pack_into, proxied_frozenset.unpack_from, proxied_frozenset),
    }

    del p

    @classmethod
    def pack_into(cls, obj, buf, offs, idmap = None, implicit_offs = 0):
        """
        Packs an :term:`RTTI`-wrapped value into ``buf`` at offset ``offs``.

        :param obj: The object to be packed. Must be one of the supported types, or an instance
            of a type registered with :meth:`register_schema`.

        :param buf: A writeable buffer onto which the object should be packed. Like a bytearray.

        :param offs: The offset within ``buf`` where to place the object.

        :param idmap: *(optional)* A mapping (dict-like or an instance of :class:`StrongIdMap`) used to
            deduplicate references to recurring objects. If not given, a temporary :class:`StrongIdMap` will
            be constructed for the operation if necessary. See :term:`idmap`.

        :param implicit_offs: *(optional)* The implicit offset of ``buf`` within a larger data structure.
            If either ``buf`` is a slice of a larger buffer or if its contents will be copied onto a larger
            buffer, this should be the starting point of ``buf``, so new entries on the ``idmap`` are
            created with the proper absolute offset. Otherwise, :term:`idmap` mixups are likely to corrupt the
            resulting buffer.

        :return: The offset where writing finished. Further objects can be placed at this offset when
            packing multiple instances.
        """
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
            cpacker.pack_into(buf, offs, typecode.encode())
            endp += cpacker.size + cpadding
            packer = cls.OBJ_PACKERS[typecode][0]
            endp = packer(obj.value, buf, endp, idmap, implicit_offs)
        else:
            raise TypeError("Unsupported type %r: %r" % (typecode, obj.value))
        return endp

    @classmethod
    @cython.locals(cpadding=cython.size_t, cpacker_size=cython.size_t, offs=cython.size_t,
        unpacker_info=tuple)
    def unpack_from(cls, buf, offs, idmap = None, proxy_into = None):
        """
        Unpacks an :term:`RTTI`-wrapped value from ``buf`` at offset ``offs``.

        :param buf: The buffer where the value is.

        :param offs: The position within ``buf`` where the value starts.

        :param idmap: *(optional)* A dict-like mapping that will be used to store references to already-unpacked
            objects, to preserve object identity (ie: to unpack the same offset into a reference to the same object).
            If object identity matters, be it for memory efficiency or some other reason, one should be provided.
            Otherwise it's not necessary for unpacking. See :term:`idmap`.

        :param proxy_into: *(optional)* An instance of :class:`BufferProxyObject` that will be turned into
            a proxy of the right kind, pointing at the right offset. This is slightly faster than building a new
            proxy every time. Otherwise a new proxy is constructed and returned.

        :return: The unpacked object or proxy, depending on the type.
        """
        cpadding = 7
        cpacker_size = 1
        buf = _likerobuffer(buf)
        typecode = chr(buf[offs])

        unpacker_info = _mapped_object_PACKERS.get(typecode)
        if unpacker_info is not None:
            packer, padding = unpacker_info
            typecode, value = packer.unpack_from(buf, offs)
            return value

        unpacker_info = _mapped_object_OBJ_PACKERS.get(typecode)
        if unpacker_info is not None:
            offs += cpacker_size + cpadding
            if proxy_into is not None:
                typ = unpacker_info[2]
                if typ is _mapped_object or type(typ) is mapped_object_with_schema:
                    return unpacker_info[1](buf, offs, idmap, proxy_into)
                else:
                    return unpacker_info[1](buf, offs, idmap)
            else:
                return unpacker_info[1](buf, offs, idmap)
        else:
            raise ValueError("Inconsistent data")

    @classmethod
    def register_schema(cls, typ, schema, typecode):
        """
        Registers the :class:`Schema` of instances of type ``typ`` and assigns it
        a typecode for :term:`RTTI`. This makes objects of those types able to be wrapped in
        :term:`RTTI` and thus referenceable as dynamically typed values, like from within
        container structures (lists, dicts, etc).

        :param typ: The type (python class) of objects represented with ``schema``.

        :param Schema schema: A :class:`Schema` that describes the shape of objects of type ``typ``.

        :param bytes typecode: A globally unique typecode (a single char) that will idenfity this type. If
            incompatible types are registered under the same typecode, an error will be raised. The typecode
            needs not be ASCII, it can be any byte, as long as it's unused. Several bytes are used by
            built-in types, see :attr:`TYPE_CODES`. The region above 127 (``'\x80'`` and beyond) will be
            explicitly reserved for used types, so it is guaranteed to be unused by built-in types.

        :rtype: mapped_object_with_schema
        :return: An instance of :class:`mapped_obect_with_schema` that can be used to pack
            instances of the registered type.
        """
        if typecode is not None:
            if typ in cls.TYPE_CODES:
                if cls.TYPE_CODES[typ] != typecode or cls.OBJ_PACKERS[typecode][2].schema is not schema:
                    raise ValueError("Registering different types with same typecode %r: %r vs %r" % (
                        typecode, cls.TYPE_CODES[typ], typ))
                return cls.OBJ_PACKERS[typecode][2]
            elif typecode in cls.PACKERS:
                raise ValueError("Registering type %r with typecode %r conflicts with builtin type" % (
                    typ, typecode))
            elif typecode in cls.OBJ_PACKERS and getattr(cls.OBJ_PACKERS[typecode][2], 'schema', None) is not schema:
                # This case is different from the first one in that, since the type isn't in TYPE_CODES,
                # it may refer to a totally different kind of type. For instance, a user-defined schema vs
                # a builtin container type, so we can't assume we have an associated schema in OBJ_PACKERS.
                raise ValueError("Registering type %r with typecode %r conflicts with previously registered type %r" % (
                    typ, typecode, cls.OBJ_PACKERS[typecode][2]))

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

    @classmethod
    def register_subclass(cls, typ, supertyp):
        """
        Registers type ``typ`` as a subclass of ``supertyp``, which means it will be treated
        as ``supertyp`` for packing/unpacking purposes. Since :term:`RTTI` will encode the supertype,
        returned proxies will be of the supertype, not the subtype, something that may need
        to be accounted for in client code.

        :rtype: mapped_object_with_schema
        :return: An instance of :class:`mapped_obect_with_schema` that can be used to pack
            instances of the registered type.
        """
        if supertyp not in cls.TYPE_CODES:
            raise ValueError("Superclass not registered: %r" % (supertyp,))

        typecode = cls.TYPE_CODES[supertyp]
        cls.TYPE_CODES[typ] = typecode
        PROXY_TYPES[typ] = PROXY_TYPES[supertyp]
        return cls.OBJ_PACKERS[typecode][2]

    @cython.locals(ivalue = cython.longlong)
    def __init__(self, value = None, typ = None):
        if value is None:
            self.typecode = None
            self.value = None
            return

        if typ is None:
            typ = type(value)

        self.value = value

        if typ is int or typ is long:
            # Special case for numbers, to pick a smaller typecode when possible
            try:
                ivalue = value
            except OverflowError:
                if 0 <= value <= 0xffffffffffffffff:
                    self.typecode = 'Q'
                    return
            else:
                if -0x80 <= ivalue <= 0x7f:
                    self.typecode = 'b'
                    return
                elif 0 <= ivalue <= 0xff:
                    self.typecode = 'B'
                    return
                elif -0x8000 <= ivalue <= 0x7fff:
                    self.typecode = 'h'
                    return
                elif 0 <= ivalue <= 0xffff:
                    self.typecode = 'H'
                    return
                elif -0x80000000 <= ivalue <= 0x7fffffff:
                    self.typecode = 'i'
                    return
                elif 0 <= ivalue <= 0xffffffff:
                    self.typecode = 'I'
                    return
                else:
                    self.typecode = 'q'
                    return

        typ = TYPES.get(typ, typ)
        if typ not in _mapped_object_TYPE_CODES and issubclass(typ, BufferProxyObject):
            # Check bases
            for base in typ.__bases__:
                if base is not object and base in _mapped_object_TYPE_CODES:
                    typ = base
                    break
        self.typecode = _mapped_object_TYPE_CODES[typ]
        self.value = value
mapped_object.TYPE_CODES[mapped_object] = 'o'
mapped_object.OBJ_PACKERS['o'] = (mapped_object.pack_into, mapped_object.unpack_from, mapped_object)

cython.declare(
    _mapped_object = object,
    _mapped_object_PACKERS = dict,
    _mapped_object_OBJ_PACKERS = dict,
    _mapped_object_TYPE_CODES = dict,
    _mapped_object_unpack_from = object,
)

_mapped_object = mapped_object
_mapped_object_PACKERS = _mapped_object.PACKERS
_mapped_object_OBJ_PACKERS = _mapped_object.OBJ_PACKERS
_mapped_object_TYPE_CODES = _mapped_object.TYPE_CODES
_mapped_object_unpack_from = _mapped_object.unpack_from

VARIABLE_TYPES = {
    frozenset : mapped_frozenset,
    set : mapped_frozenset,
    tuple : mapped_tuple,
    list : mapped_list,
    dict : mapped_dict,
    collections.defaultdict : mapped_dict,
    str : mapped_bytes,
    six.text_type : mapped_unicode,
    bytes : mapped_bytes,
    object : mapped_object,
    datetime : mapped_datetime,
    date : mapped_date,
    Decimal : mapped_decimal,
    cDecimal : mapped_decimal,
    numpy.ndarray : proxied_ndarray,
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
    for v in itervalues(VARIABLE_TYPES)
})
del t

@cython.cclass
class BufferProxyObject(object):
    """
    A base class for object proxies.

    See :func:`GenericProxyClass` for a higher level way to construct these.
    """

    cython.declare(
        __weakref__ = object,
        buf = object,
        idmap = object,
        pybuf = 'Py_buffer',
        offs = cython.Py_ssize_t,
        none_bitmap = cython.ulonglong
    )

    @property
    def _proxy_buffer(self):
        return self.buf

    @property
    def _proxy_offset(self):
        return self.offs

    def __cinit__(self, buf, offs, none_bitmap, idmap = None):
        if cython.compiled:
            self.pybuf.buf = cython.NULL

    @cython.locals(offs = cython.Py_ssize_t, none_bitmap = cython.ulonglong)
    def __init__(self, buf, offs, none_bitmap, idmap = None):
        self._init_internal(buf, offs, none_bitmap, idmap)

    @cython.cfunc
    @cython.locals(offs = cython.Py_ssize_t, none_bitmap = cython.ulonglong)
    def _init_internal(self, buf, offs, none_bitmap, idmap):
        if cython.compiled:
            if self.pybuf.buf != cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok
                self.pybuf.buf = cython.NULL

        self.buf = buf
        self.idmap = idmap
        self.offs = offs
        self.none_bitmap = none_bitmap

        if cython.compiled:
            try:
                PyObject_GetBuffer(buf, cython.address(self.pybuf), PyBUF_WRITABLE)  # lint:ok
            except BufferError:
                PyObject_GetBuffer(buf, cython.address(self.pybuf), PyBUF_SIMPLE)  # lint:ok

    @cython.ccall
    @cython.locals(offs = cython.Py_ssize_t, none_bitmap = cython.ulonglong)
    def _init(self, buf, offs, none_bitmap, idmap):
        self._init_internal(buf, offs, none_bitmap, idmap)

    @cython.cfunc
    @cython.locals(offs = cython.Py_ssize_t, none_bitmap = cython.ulonglong)
    def _reset_internal(self, offs, none_bitmap, idmap):
        self.offs = offs
        self.none_bitmap = none_bitmap
        self.idmap = idmap

    @cython.ccall
    @cython.locals(offs = cython.Py_ssize_t, none_bitmap = cython.ulonglong)
    def _reset(self, offs, none_bitmap, idmap):
        self._reset_internal(offs, none_bitmap, idmap)

    def __dealloc__(self):
        if cython.compiled:
            if self.pybuf.buf != cython.NULL:
                PyBuffer_Release(cython.address(self.pybuf))  # lint:ok
                self.pybuf.buf = cython.NULL

@cython.cclass
class BaseBufferProxyProperty(object):
    cython.declare(offs = cython.Py_ssize_t, mask = cython.ulonglong)

    def __init__(self, offs, mask):
        self.offs = offs
        self.mask = mask
        self._init_impl()

    @cython.cfunc
    def _init_impl(self):
        pass

    def __set__(self, obj, value):
        raise TypeError("Proxy objects are read-only")

    def cas(self, obj, exp_val, new_val):
        raise TypeError("Proxy objects are read-only")

    def __delete__(self, obj):
        raise TypeError("Proxy objects are read-only")


if cython.compiled:
    # We need as many of these definitions as different parameters are
    # used per-function; otherwise, cython will deduce that they are of
    # the same type. Sigh ...

    numeric_A = cython.fused_type(
        cython.schar,
        cython.uchar,
        cython.sshort,
        cython.ushort,
        cython.sint,
        cython.uint,
        cython.slong,
        cython.ulong,
        cython.slonglong,
        cython.ulonglong,
        cython.float,
        cython.double,
    )

    numeric_B = cython.fused_type(
        cython.schar,
        cython.uchar,
        cython.sshort,
        cython.ushort,
        cython.sint,
        cython.uint,
        cython.slong,
        cython.ulong,
        cython.slonglong,
        cython.ulonglong,
        cython.float,
        cython.double,
    )
else:
    globals().update(dict(
        numeric_A = object,
        numeric_B = object,
    ))


if cython.compiled:
    @cython.cfunc
    @cython.locals(self = BaseBufferProxyProperty, elem = numeric_A, obj = BufferProxyObject,
        ptr = 'numeric_A *')
    def _c_buffer_proxy_get_gen(self, obj, elem):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        assert (obj.offs + self.offs + cython.sizeof(elem)) <= obj.pybuf.len   #lint:ok
        ptr = cython.cast('numeric_A *',
            cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)   #lint: ok
        if not obj.pybuf.readonly:
            mfence_full()   # acquire
        return ptr[0]

    @cython.cfunc
    @cython.locals(self = BaseBufferProxyProperty, elem = numeric_A, obj = BufferProxyObject)
    def _c_buffer_proxy_set_gen(self, obj, elem):
        if obj is None or (obj.none_bitmap & self.mask):
            return
        elif obj.pybuf.readonly:
            raise TypeError('cannot set attribute in read-only buffer')
        assert (obj.offs + self.offs + cython.sizeof(elem)) <= obj.pybuf.len   #lint:ok
        cython.cast('numeric_A *',
            cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0] = elem   #lint:ok
        mfence_full()   # release

    @cython.cfunc
    @cython.inline
    @cython.locals(self = BaseBufferProxyProperty, obj = BufferProxyObject,
        exp_val = numeric_A, new_val = numeric_A, ptr = 'numeric_A *')
    def _c_buffer_proxy_atomic_cas(self, obj, exp_val, new_val):
        if obj is None or (obj.none_bitmap & self.mask):
            return False
        elif obj.pybuf.readonly:
            raise TypeError('cannot set attribute in read-only buffer')
        assert (obj.offs + self.offs + cython.sizeof(exp_val)) <= obj.pybuf.len   #lint:ok
        ptr = cython.cast('numeric_A *',
            cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)
        if numeric_A is cython.float:
            return _c_atomic_cas_flt(ptr, exp_val, new_val)
        elif numeric_A is cython.double:
            return _c_atomic_cas_dbl(ptr, exp_val, new_val)
        else:
            return _c_atomic_cas(ptr, exp_val, new_val)

    @cython.cfunc
    @cython.inline
    @cython.locals(self = BaseBufferProxyProperty, obj = BufferProxyObject,
        value = numeric_A, ptr = 'numeric_A *')
    def _c_buffer_proxy_atomic_add(self, obj, value):
        if obj is None or (obj.none_bitmap & self.mask):
            return
        elif obj.pybuf.readonly:
            raise TypeError('cannot set attribute in read-only buffer')
        assert (obj.offs + self.offs + cython.sizeof(value)) <= obj.pybuf.len   #lint:ok
        ptr = cython.cast('numeric_A *',
            cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)
        if numeric_A is cython.float:
            _c_atomic_add_flt(ptr, value)
        elif numeric_A is cython.double:
            _c_atomic_add_dbl(ptr, value)
        else:
            _c_atomic_add(ptr, value)

else:
    def _buffer_proxy_get(self, obj, code):
        if obj is None:
            return self
        elif obj.none_bitmap & self.mask:
            return None
        else:
            return struct.unpack_from(code, obj.buf, obj.offs + self.offs)[0]

    def _buffer_proxy_set(self, obj, code, elem):
        if obj is not None and not (obj.none_bitmap & self.mask):
            struct.pack_into(code, obj.buf, obj.offs + self.offs, elem)

    def _buffer_proxy_cas(self, obj, code, exp_val, new_val):
        # XXX: This is not atomic!
        if obj is not None and not (obj.none_bitmap & self.mask):
            tmp = struct.unpack_from(code, obj.buf, obj.offs + self.offs)[0]
            if tmp == exp_val:
                struct.pack_into(code, obj.buf, obj.offs + self.offs, new_val)
                return True
        return False

    def _buffer_proxy_add(self, obj, code, value):
        if obj is not None and not (obj.none_bitmap & self.mask):
            tmp = struct.unpack_from(code, obj.buf, obj.offs + self.offs)[0]
            struct.pack_into(code, obj.buf, obj.offs + self.offs, tmp + value)


@cython.cclass
class BoolBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar) if cython.compiled else struct.Struct('B').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.uchar](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'B')

    @cython.locals(obj = BufferProxyObject, elem = cython.uchar)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.uchar](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'B', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.uchar, new_val = cython.uchar)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.uchar](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'B', exp_val, new_val)


@cython.cclass
class UByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uchar) if cython.compiled else struct.Struct('B').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.uchar](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'B')

    @cython.locals(obj = BufferProxyObject, elem = cython.uchar)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.uchar](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'B', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.uchar, new_val = cython.uchar)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.uchar](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'B', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.uchar)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.uchar](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'B', value)


@cython.cclass
class ByteBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.schar) if cython.compiled else struct.Struct('b').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.schar](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'b')

    @cython.locals(obj = BufferProxyObject, elem = cython.schar)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.schar](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'b', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.schar, new_val = cython.schar)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.schar](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'b', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.schar)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.schar](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'b', value)

@cython.cclass
class UShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.ushort) if cython.compiled else struct.Struct('H').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.ushort](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'H')

    @cython.locals(obj = BufferProxyObject, elem = cython.ushort)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.ushort](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'H', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.ushort, new_val = cython.ushort)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.ushort](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'H', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.ushort)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.ushort](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'H', value)

@cython.cclass
class ShortBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.sshort) if cython.compiled else struct.Struct('h').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.sshort](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'h')

    @cython.locals(obj = BufferProxyObject, elem = cython.sshort)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.sshort](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'h', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.sshort, new_val = cython.sshort)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.sshort](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'h', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.sshort)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.sshort](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'h', value)

@cython.cclass
class UIntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.uint) if cython.compiled else struct.Struct('I').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.uint](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'I')

    @cython.locals(obj = BufferProxyObject, elem = cython.uint)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.uint](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'I', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.uint, new_val = cython.uint)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.uint](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'I', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.uint)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.uint](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'I', value)

@cython.cclass
class IntBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.int) if cython.compiled else struct.Struct('i').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.sint](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'i')

    @cython.locals(obj = BufferProxyObject, elem = cython.int)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.sint](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'i', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.sint, new_val = cython.sint)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.sint](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'i', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.sint)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.sint](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'i', value)

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
            rv = cython.cast(cython.p_ulonglong,
                cython.cast(cython.p_uchar, obj.pybuf.buf) + obj.offs + self.offs)[0]  # lint:ok
            if rv <= cython.cast(cython.ulonglong, 0x7FFFFFFFFFFFFFFF):
                return cython.cast(cython.longlong, rv)
            else:
                return rv
        else:
            return struct.unpack_from('Q', obj.buf, obj.offs + self.offs)[0]

    @cython.locals(obj = BufferProxyObject, elem = cython.ulonglong)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.ulonglong](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'Q', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.ulonglong, new_val = cython.ulonglong)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.ulonglong](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'Q', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.ulonglong)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.ulonglong](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'Q', value)

@cython.cclass
class LongBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.slonglong](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'q')

    @cython.locals(obj = BufferProxyObject, elem = cython.slonglong)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.slonglong](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'q', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.longlong, new_val = cython.longlong)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.slonglong](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'q', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.slonglong)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.slonglong](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'q', value)

@cython.cclass
class DoubleBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.double) if cython.compiled else struct.Struct('d').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.double](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'd')

    @cython.locals(obj = BufferProxyObject, elem = cython.double)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.double](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'd', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.double, new_val = cython.double)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.double](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'd', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.double)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.double](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'd', value)

@cython.cclass
class FloatBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.float) if cython.compiled else struct.Struct('f').size

    @cython.locals(obj = BufferProxyObject)
    def __get__(self, obj, klass):
        if cython.compiled:
            return _c_buffer_proxy_get_gen[cython.float](self, obj, 0)
        else:
            return _buffer_proxy_get(self, obj, 'f')

    @cython.locals(obj = BufferProxyObject, elem = cython.float)
    def __set__(self, obj, elem):
        if cython.compiled:
            _c_buffer_proxy_set_gen[cython.float](self, obj, elem)
        else:
            _buffer_proxy_set(self, obj, 'f', elem)

    @cython.locals(obj = BufferProxyObject, exp_val = cython.float, new_val = cython.float)
    def cas(self, obj, exp_val, new_val):
        if cython.compiled:
            return _c_buffer_proxy_atomic_cas[cython.float](self, obj, exp_val, new_val)
        else:
            return _buffer_proxy_cas(self, obj, 'f', exp_val, new_val)

    @cython.locals(obj = BufferProxyObject, value = cython.float)
    def add(self, obj, value):
        if cython.compiled:
            _c_buffer_proxy_atomic_add[cython.float](self, obj, value)
        else:
            _buffer_proxy_add(self, obj, 'f', value)

@cython.cclass
class BytesBufferProxyProperty(BaseBufferProxyProperty):
    stride = cython.sizeof(cython.longlong) if cython.compiled else struct.Struct('q').size

    @cython.locals(obj = BufferProxyObject, offs = cython.Py_ssize_t, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
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
                if type(obj.idmap) is dict:
                    rv = cython.cast(dict, obj.idmap).get(poffs)
                else:
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

    @cython.locals(obj = BufferProxyObject, offs = cython.Py_ssize_t, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
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
                if type(obj.idmap) is dict:
                    rv = cython.cast(dict, obj.idmap).get(poffs)
                else:
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

    cython.declare(
        _unpack_from = object,
    )

    @cython.cfunc
    def _init_impl(self):
        self._unpack_from = self.typ.unpack_from

    @cython.locals(obj = BufferProxyObject, offs = cython.Py_ssize_t, buflen = cython.ulonglong, pybuf = 'Py_buffer*',
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
                if type(obj.idmap) is dict:
                    rv = cython.cast(dict, obj.idmap).get(poffs, poffs)
                else:
                    rv = obj.idmap.get(poffs, poffs)
                # idmap cannot possibly hold "poffs" for that offset
                if rv is not poffs:
                    return rv
            assert offs + cython.sizeof(cython.ushort) <= buflen
        else:
            poffs = offs = obj.offs + struct.unpack_from('q', obj.buf, obj.offs + self.offs)[0]
        rv = self._unpack_from(obj.buf, offs)
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

@cython.cclass
class ProxiedFrozensetBufferProxyProperty(GenericBufferProxyProperty):
    typ = proxied_frozenset

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
    proxied_frozenset : ProxiedFrozensetBufferProxyProperty,

    int : LongBufferProxyProperty,
    long : LongBufferProxyProperty,
    float : DoubleBufferProxyProperty,
    str : BytesBufferProxyProperty,
    six.text_type : UnicodeBufferProxyProperty,
    datetime : DatetimeBufferProxyProperty,
    date : DateBufferProxyProperty,
    Decimal : DecimalBufferProxyProperty,
    cDecimal : DecimalBufferProxyProperty,
    numpy.ndarray : ProxiedNDArrayBufferProxyProperty,
    buffer : ProxiedBufferBufferProxyProperty,
}

def GenericProxyClass(slot_keys, slot_types, present_bitmap, base_offs, bases = None):
    """
    Construct a subclass of :class:`BufferProxyObject` with the given shape.

    Since the shape depends on the attributes that are present with an explicit value,
    a different proxy type will be necessary for values with different sets of missing/null
    attributes.

    :type slot_keys: sequence[str]
    :param slot_keys: An ordered list of attribute names.

    :type slot_types: sequence[type]
    :param slot_types: For each attribute in ``slot_keys``, its type

    :param int present_bitmap: A bitmap specifying which attributes are actually present.
        That is, those that have an explicit not-None value.

    :param int base_offs: The offset of the first attribute's value relative to the object's
        initial offset.

    :type bases: tuple or None
    :param bases: *(optional)* If given, the returned class will inherit from ``bases``.

    :return: Subtype of :class:`BufferProxyObject` implementing the specified attributes
        through readable data properties.
    """
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

cython.declare(_GenericProxy = object)
GenericProxy = _GenericProxy = GenericProxyClass([], {}, 0, 0)
_GenericProxy_new = cython.declare(object, _GenericProxy.__new__)

@cython.cclass
class Schema(object):
    """
    A declaration of an object's shape, or *schema*.

    A schema is constructed out of a mapping of attribute names to attribute types
    with :meth:`from_typed_slots`.

    After construction, the schema instance can be used to :meth:`pack_into` and
    :meth:`unpack_from` shared memory buffers. When unpacking, proxies will be
    created automatically based on the underlying object's shape.

    Schemas are picklable, so you may store the schema alongside the buffer itself
    to make portable buffers that should be compatible across versions, as long as
    the data itself is compatible (ie: no missing attributes that the application
    really needs).

    Proxies can be made to inherit from a base class or classes by using :meth:`set_proxy_bases`.
    This can be useful if the objects stored in the shared buffer have behavior that the
    proxies should also exhibit.
    """

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
        fast_unpacker_cache = dict,
        fast_unpacker_cache_size = cython.size_t,

        prewrite_hook = object,
        postwrite_hook = object,

        _Proxy = object,
        _proxy_bases = tuple,

        _pack_buffer = object,
        _var_bitmap = cython.ulonglong,
        _fixed_bitmap = cython.ulonglong,
        _last_unpacker = tuple,
        _last_unpacker_bitmap = cython.ulonglong,
        _binary_version = cython.uint,
    )

    VERSION = 2

    @property
    def Proxy(self):
        """
        A factory callable that constructs proxies of the right kind, pointing to nowhere.
        The caller is expected to :meth:`~BufferProxyObject._init` the proxy and repoint
        it somewhere useful after building it.
        """
        return functools.partial(self._Proxy, b"\x00" * self.bitmap_size, 0, 0, None)

    @property
    def ProxyClass(self):
        """
        Returns the type of returned proxies, a subtype of :class:`BufferProxyObject` generated
        with :func:`GenericProxyClass`.
        """
        return self._Proxy

    def __init__(self, slot_types, alignment = 8, pack_buffer_size = 65536, packer_cache = None, unpacker_cache = None,
            max_pack_buffer_size = None, version = None):
        self.prewrite_hook = None
        self.postwrite_hook = None
        self.init(
            self._map_types(slot_types),
            packer_cache = packer_cache, unpacker_cache = unpacker_cache, alignment = alignment,
            pack_buffer_size = pack_buffer_size,
            version = self.VERSION)

    def __reduce__(self):
        return (type(self), (self.slot_types,), self.__getstate__())

    def __getstate__(self):
        return dict(
            slot_types = self.slot_types,
            slot_keys = self.slot_keys,
            alignment = self.alignment,
            bases = self._proxy_bases,
            version = self._binary_version,
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
        """
        Sets the bases of future constructed proxies.
        It should be called before unpacking objects, or wrong proxy classes could
        remain in the proxy cache for quite a while.

        :param tuple bases: The base classes of constructed proxies.
        """
        self._proxy_bases = bases

    def set_prewrite_hook(self, hook):
        """
        A callable that will be invoked with arguments (obj, buf, baseoffs)
        each time an object is about to be packed into a buffer.
        """
        self.prewrite_hook = hook

    def set_postwrite_hook(self, hook):
        """
        A callable that will be invoked with arguments (obj, buf, baseoffs, offs)
        each time after an object has been packed into a buffer.
        """
        self.postwrite_hook = hook

    @cython.locals(other_schema = 'Schema')
    def compatible(self, other):
        """
        Checks whether this schema is compatible with ``other``.

        Compatible schemas are those that have binary-compatible in-buffer data representations.
        """
        if not isinstance(other, Schema):
            return False

        other_schema = other
        if self._binary_version != other_schema._binary_version:
            return False

        if self.slot_keys != other_schema.slot_keys or self.alignment != other_schema.alignment:
            return False

        for k in self.slot_keys:
            if self.slot_types[k] is not other_schema.slot_types[k]:
                return False

        return True

    @staticmethod
    def _map_types(slot_types):
        return { k:TYPES.get(v,v) for k,v in iteritems(slot_types) }

    @classmethod
    def from_typed_slots(cls, struct_class_or_slot_types, *p, **kw):
        """
        Constructs a :class:`Schema` out of a description of attribute (slot) types.

        :param struct_class_or_slot_types: Either a class with a ``__slot_types__`` attribute,
            or the mapping of attribute names to attribute types itself.

        :keyword int alignment: *(default 8)* Enforce alignment on object sizes. Having objects
            aligned to a native word size helps with performance.

        :keyword int pack_buffer_size: *(default 64k)* Initial :meth:`pack` buffer size. Will auto-expand when
            necessary, but having it sized correctly from the start can help avoid the performance impact of
            such resizing.

        :keyword int max_pack_buffer_size: *(optional)* Maximum :meth:`pack` buffer size. Will not auto-expand
            beyond this.

        :rtype: Schema
        """
        if hasattr(struct_class_or_slot_types, '__slot_types__'):
            return cls(struct_class_or_slot_types.__slot_types__, *p, **kw)
        elif isinstance(struct_class_or_slot_types, dict):
            return cls(struct_class_or_slot_types, *p, **kw)
        else:
            raise ValueError("Cant build a schema out of %r" % (type(struct_class_or_slot_types),))

    def set_pack_buffer_size(self, newsize):
        """
        Sets a new :meth:`pack` buffer size. See :meth:`from_typed_slots`.
        """
        self.pack_buffer_size = newsize
        self._pack_buffer = bytearray(self.pack_buffer_size)

    def reinitialize(self):
        """
        Reinitialize schema to account for newly registered types. Call after registering
        all required types if you can't register them beforehand.
        """
        self.init(
            self._map_types(self.slot_types),
            packer_cache = self.packer_cache, unpacker_cache = self.unpacker_cache, alignment = self.alignment,
            pack_buffer_size = self.pack_buffer_size)

    @cython.locals(slot_types = dict, slot_keys = tuple, version = cython.uint)
    def init(self, slot_types = None, slot_keys = None, alignment = 8, pack_buffer_size = 65536,
            max_pack_buffer_size = None, packer_cache = None, unpacker_cache = None,
            autoregister = False, version = 1):
        # Freeze slot order, sort by descending size to optimize alignment
        self._proxy_bases = None
        self._binary_version = version
        if slot_types is None:
            slot_types = self.slot_types

        # Map compatible types
        for k, typ in iteritems(slot_types):
            if not isinstance(typ, mapped_object_with_schema):
                continue
            if typ in mapped_object.TYPE_CODES:
                continue

            typ_schema = getattr(typ, 'schema', None)
            if typ_schema is None:
                continue

            # Find compatible type
            for packer, unpacker, packable_type in itervalues(mapped_object.OBJ_PACKERS):
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
        self.fast_unpacker_cache = {}
        self.fast_unpacker_cache_size = 256
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
            for slot, typ in iteritems(self.slot_types)
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

        if (self._binary_version < 2 or self.alignment <= 2) and len(self.slot_keys) <= 8:
            self.bitmap_type = 'B'
        elif (self._binary_version < 2 or self.alignment <= 4) and len(self.slot_keys) <= 16:
            self.bitmap_type = 'H'
        elif (self._binary_version < 2 or self.alignment <= 8) and len(self.slot_keys) <= 32:
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
        i = int, get_values = cython.bint)
    @cython.returns(tuple)
    def _get_bitmaps(self, obj, get_values=False):
        has_bitmap = 0
        none_bitmap = 0
        if get_values:
            values = []
        for i,slot in enumerate(self.slot_keys):
            try:
                val = getattr(obj, slot)
            except AttributeError:
                pass
            else:
                has_bitmap |= cython.cast(cython.ulonglong, 1) << i
                if val is None:
                    none_bitmap |= cython.cast(cython.ulonglong, 1) << i
                elif get_values:
                    values.append(val)
        present_bitmap = has_bitmap & ~none_bitmap
        if get_values:
            return has_bitmap, none_bitmap, present_bitmap, values
        else:
            return has_bitmap, none_bitmap, present_bitmap

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, rv = tuple, packer_key = object)
    @cython.returns(tuple)
    def get_packer(self, obj):
        has_bitmap, none_bitmap, present_bitmap = self._get_bitmaps(obj)

        if present_bitmap <= cython.cast(cython.ulonglong, 0x7FFFFFFFFFFFFFFF):
            packer_key = cython.cast(cython.longlong, present_bitmap)
        else:
            packer_key = present_bitmap

        rv = self.packer_cache.get(packer_key)
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
            padding = (size + alignment - 1) // alignment * alignment - size
            self.packer_cache[packer_key] = rv = (packer, padding)
        return rv

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, rv = tuple, unpacker_key = object)
    @cython.returns(tuple)
    def get_unpacker(self, has_bitmap, none_bitmap):
        present_bitmap = has_bitmap & ~none_bitmap
        if self._last_unpacker is not None and present_bitmap == self._last_unpacker_bitmap:
            return self._last_unpacker

        if present_bitmap <= cython.cast(cython.ulonglong, 0x7FFFFFFFFFFFFFFF):
            unpacker_key = cython.cast(cython.longlong, present_bitmap)
        else:
            unpacker_key = present_bitmap

        rv = self.fast_unpacker_cache.get(unpacker_key)
        if rv is None:
            rv = self.unpacker_cache.get(unpacker_key)
        if rv is None:
            pformat = "".join([
                self.slot_struct_types[slot]
                for i,slot in enumerate(self.slot_keys)
                if present_bitmap & (cython.cast(cython.ulonglong, 1) << i)
            ])
            unpacker = struct.Struct(pformat)
            alignment = self.alignment
            size = unpacker.size
            padding = (size + self.bitmap_size + alignment - 1) // alignment * alignment - size
            gfactory = GenericProxyClass(
                self.slot_keys, self.slot_types, present_bitmap, self.bitmap_size,
                self._proxy_bases)
            rv = (unpacker, padding, pformat, gfactory)
            self.unpacker_cache[unpacker_key] = rv
            self.fast_unpacker_cache[unpacker_key] = rv
            if len(self.fast_unpacker_cache) > self.fast_unpacker_cache_size:
                self.fast_unpacker_cache.clear()
        self._last_unpacker_bitmap = present_bitmap
        self._last_unpacker = rv
        return rv

    @cython.ccall
    @cython.locals(has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, present_bitmap = cython.ulonglong,
        i = int, size = int, alignment = int, padding = int, val_pos = int, mask = cython.ulonglong,
        offs = cython.longlong, implicit_offs = cython.longlong, ival_offs = cython.longlong,
        widmap = StrongIdMap, values = list)
    @cython.returns(tuple)
    def get_packable(self, packer, padding, obj, offs = 0, buf = None, idmap = None, implicit_offs = 0):
        if idmap is None:
            idmap = StrongIdMap()
        if isinstance(idmap, StrongIdMap):
            widmap = idmap
        else:
            widmap = None
        baseoffs = offs
        has_bitmap, none_bitmap, present_bitmap, values = self._get_bitmaps(obj, True)
        fixed_present = present_bitmap & self._fixed_bitmap
        size = packer.size
        offs += size + padding

        if buf is None:
            buf = self._acquire_pack_buffer()
            owns_buffer = True
        else:
            owns_buffer = False

        try:
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
            val_pos = 0
            for i,slot in enumerate(self.slot_keys):
                mask = cython.cast(cython.ulonglong, 1) << i
                if present_bitmap & mask:
                    val = values[val_pos]
                    val_pos += 1
                    if fixed_present & mask:
                        packable_append(val)
                    else:
                        slot_type = slot_types[slot]
                        if slot_type is _mapped_object:
                            val_id = wrapped_id(val)
                        else:
                            val_id = shared_id(val)
                        if widmap is not None:
                            # fast-call
                            val_offs = widmap.get(val_id)
                        else:
                            val_offs = idmap_get(val_id)
                        if val_offs is None:
                            idmap[val_id] = ival_offs = offs + implicit_offs
                            if widmap is not None:
                                widmap.link(val_id, val)
                            try:
                                offs = slot_type.pack_into(val, buf, offs, idmap, implicit_offs)
                            except Exception as e:
                                try:
                                    # Add some context. It may not work with all exception types, hence the fallback
                                    vrepr = repr(val)
                                    if len(vrepr) > 200:
                                        vrepr = vrepr[:200] + '...'
                                    e = type(e)("%s packing attribute %s=%s of type %s" % (
                                        e, slot, vrepr, type(obj).__name__))
                                except:
                                    pass
                                else:
                                    reraise(type(e), e, sys.exc_info()[2])
                                raise
                            padding = (offs + alignment - 1) // alignment * alignment - offs
                            offs += padding
                        else:
                            ival_offs = val_offs
                        packable_append(ival_offs - baseoffs - implicit_offs)

            padding = (offs + alignment - 1) // alignment * alignment - offs
            offs = offs + padding
            if offs > len(buf):
                raise struct.error('buffer too small')
            return packable, offs
        finally:
            if owns_buffer:
                self._release_pack_buffer(buf)

    @cython.ccall
    def pack_into(self, obj, buf, offs, idmap = None, packer = None, padding = None, implicit_offs = 0):
        """
        Pack ``obj`` into ``buf`` at offset ``ofs``. The object just needs to have the attributes
        declared in the schema, its type is unimportant.

        See :meth:`mapped_object.pack_into` for a description of the arguments.
        """
        if idmap is None:
            idmap = StrongIdMap()
        if packer is None:
            packer, padding = self.get_packer(obj)
        baseoffs = offs
        packable, offs = self.get_packable(packer, padding, obj, offs, buf, idmap, implicit_offs)
        if self.prewrite_hook is not None:
            self.prewrite_hook(obj, buf, offs)
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
        if self.postwrite_hook is not None:
            self.postwrite_hook(obj, buf, baseoffs, offs)
        return offs

    @cython.cfunc
    def _acquire_pack_buffer(self):
        pack_buffer = self._pack_buffer
        if pack_buffer is None:
            pack_buffer = bytearray(self.pack_buffer_size)
        else:
            self._pack_buffer = None
        return pack_buffer

    @cython.cfunc
    def _release_pack_buffer(self, pack_buffer):
        if self._pack_buffer is None and pack_buffer is not None:
            self._pack_buffer = pack_buffer

    @cython.ccall
    def pack(self, obj, idmap = None, packer = None, padding = None, implicit_offs = 0):
        """
        Pack ``obj`` into a byte array and return the corresponding slice.

        See :meth:`pack_into` for a description of the arguments.
        """
        buf = self._acquire_pack_buffer()
        try:
            for i in range(24):
                try:
                    endp = self.pack_into(obj, buf, 0, idmap, packer, padding, implicit_offs)
                    return buf[:endp]
                except (struct.error, IndexError):
                    # Buffer overflow, retry with a bigger buffer
                    # Idmap is probably corrupted beyond hope though :(
                    if len(buf) >= self.max_pack_buffer_size:
                        raise
                    buf.extend(buf)
                    if idmap is not None:
                        idmap.clear()
        finally:
            self._release_pack_buffer(buf)

    @cython.ccall
    @cython.locals(
        offs = cython.longlong, padding = cython.longlong, baseoffs = cython.longlong, i = int, value_ix = int,
        has_bitmap = cython.ulonglong, none_bitmap = cython.ulonglong, values = tuple, stride = cython.longlong,
        proxy_into = BufferProxyObject,
        pbuf = 'char *', pbuf2 = 'char *', pformat = 'char *', formatchar = 'char', pybuf='Py_buffer')
    def unpack_from(self, buf, offs = 0, idmap = None, factory_class_new = None, proxy_into = None):
        """
        Unpack data from ``buf`` at offset ``ofs``.

        See :meth:`mapped_object.unpack_from` for a description of the arguments.
        """
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
                    if type(proxy_into) is not gfactory:
                        proxy_into.__class__ = gfactory
                    if proxy_into.buf is buf:
                        proxy_into._reset_internal(offs, none_bitmap, idmap)
                    else:
                        proxy_into._init_internal(buf, offs, none_bitmap, idmap)
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
                    for i in range(self.slot_count):
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
                                    value = cython.cast(cython.p_schar, pbuf)[0]
                                    pbuf += cython.sizeof(cython.schar)
                                elif formatchar == '?':
                                    value = cython.cast(cython.bint, cython.cast(cython.p_char, pbuf)[0])
                                    pbuf += cython.sizeof(cython.char)
                                elif formatchar == 'H':
                                    value = cython.cast(cython.p_ushort, pbuf)[0]
                                    pbuf += cython.sizeof(cython.ushort)
                                elif formatchar == 'h':
                                    value = cython.cast(cython.p_short, pbuf)[0]
                                    pbuf += cython.sizeof(cython.sshort)
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
                    for i in range(self.slot_count):
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
        """
        Unpack data from ``buf``.

        See :meth:`unpack_from` for a description of the arguments.
        """
        return self.unpack_from(buffer(buf), 0, idmap, factory_class_new, proxy_into)

@cython.cclass
class mapped_object_with_schema(object):
    """
    An object that can be used to pack and unpack objects with the given :class:`Schema`.

    Used mostly internally for schema pickling.
    """

    cython.declare(_schema = Schema)

    def __init__(self, schema):
        self._schema = schema

    @property
    def schema(self):
        """
        The objects' :class:`Schema`
        """
        return self._schema

    def pack_into(self, obj, buf, offs, idmap = None, implicit_offs = 0):
        """
        Packs ``obj`` with :attr:`schema`.

        See also :meth:`mapped_object.pack_into`
        """
        return self._schema.pack_into(obj, buf, offs, idmap, None, None, implicit_offs)

    def unpack_from(self, buf, offs, idmap = None, proxy_into = None):
        """
        Unpacks ``obj`` with :attr:`schema`.

        See also :meth:`mapped_object.pack_into`
        """
        return self._schema.unpack_from(buf, offs, idmap, None, proxy_into)

    def __reduce__(self):
        # WARNING: Not using setstate somehow breaks schemas with cyclic references \_(0_0)_/
        return (type(self), (None,), (self._schema,))

    def __setstate__(self, state):
        if isinstance(state, (tuple, list)):
            self._schema = state[0]
        elif isinstance(state, dict):
            self._schema = state['schema']
        else:
            raise ValueError("Bad mapped_object_with_schema state: %r" % (state,))

def __pyx_unpickle_mapped_object_with_schema(__pyx_type, __pyx_checksum, __pyx_state):
    # For compatibility with older pickles only
    result = mapped_object_with_schema.__new__(__pyx_type)
    if __pyx_state is not None:
        result.__setstate__(__pyx_state)
    return result

@cython.ccall
def _map_zipfile(cls, fileobj, offset, size, read_only):
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

    return cls.map_file(fileobj._fileobj, offset, size, read_only)

class _ZipMapBase(object):
    @classmethod
    def map_zipfile(cls, fileobj, offset = 0, size = None, read_only = True):
        return _map_zipfile(cls, fileobj, offset, size, read_only)

@cython.cclass
class _CZipMapBase(object):
    @classmethod
    def map_zipfile(cls, fileobj, offset = 0, size = None, read_only = True):
        return _map_zipfile(cls, fileobj, offset, size, read_only)

class GenericFileMapper(_ZipMapBase):
    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        Returns a buffer mapping the file object's requested
        range, and the underlying mmap object as a tuple.
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        if size is None:
            fileobj.seek(0, os.SEEK_END)
            size = fileobj.tell() - offset
        fileobj.seek(offset)
        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        if read_only:
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE
        buf = mmap.mmap(fileobj.fileno(), size + offset - map_start,
            access = access, offset = map_start)
        return buffer(buf, offset - map_start, size), buf

class MappedArrayProxyBase(_ZipMapBase):
    """
    Base class for arrays of objects with a uniform :class:`Schema`.

    Construct a concrete class by subclassing and providing a :class:`Schema`::

        class SomeArrayType(MappedArrayProxyBase):
            schema = Schema.from_typed_slots(SomeClass)

    Then build them into temporary files by using :meth:`build`::

        mapped_array = SomeArrayType.build(iterable)

    The returned array will be mapped from a temporary file. You can also provide
    an explicit file where to build the array instead. See :meth:`build` for details.

    The schema is pickled into the buffer so the array should be portable.

    The array class implements the (readonly) sequence interface, supporting iteration,
    length, random access subscripting, but not slicing.
    """

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
        self.wr_buf = buf   # May be overriden by map_file
        self.total_size, self.index_offset, self.index_elements = self._Header.unpack_from(buf, 0)
        self.index = npfrombuffer(buf,
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

    @cython.locals(schema = Schema, proxy_into = BufferProxyObject, read_only = cython.bint)
    def getter(self, proxy_into = None, no_idmap = False):
        """
        Build a getter callable to quickly access items in succession.

        Building a getter instead of using subscript syntax can provide a performance
        boost, especially when specifying ``proxy_into`` to reuse proxies instead
        of building a new one at each invocation.

        When reusing proxies, have in mind that after a call to the getter,
        any existing reference to the earlier proxy will be reset into the new
        object::

            g = array.getter(proxy_into=schema.Proxy())
            a = getter(1)
            b = getter(2)
            # a and b point to the same object at this point

        :param BufferProxyObject proxy_into: *(optional)* A proxy object to be reused at each invocation.

        :param bool no_idmap: *(default False)* If true, no :term:`idmap` will be used. The resulting
            getter will use less memory, but may break identity relationships, returning
            copies of objects that should be identical instead.
        """
        schema = self.schema
        proxy_class = self.proxy_class
        index = self.index
        idmap = self.idmap if not no_idmap else None
        buf = self.wr_buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None

        @cython.locals(pos = int)
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
        buf = self.wr_buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None

        for i in range(len(self)):
            yield schema.unpack_from(buf, index[i], idmap, proxy_class_new)

    @cython.locals(i = int, schema = Schema, pmask = 'const unsigned char[:]')
    def iter_fast(self, mask=None):
        """
        Iterates through the array by reusing a single proxy object instead of building a new one per item.
        See :meth:`getter`.

        :param mask: *(optional)* If given, it should be a numpy array or typed memoryview of bytes,
            with a length equal to the length of the array, with a mask flagging the items that are
            to be retrieved with a nonzero value. It implements the same semantic as indexing a numpy
            array with a boolean mask.
        """
        # getter inlined
        schema = self.schema
        proxy_class = self.proxy_class
        index = self.index
        idmap = self.idmap
        buf = self.wr_buf

        if proxy_class is not None:
            proxy_class_new = functools.partial(proxy_class.__new__, proxy_class)
        else:
            proxy_class_new = None

        if mask is not None:
            pmask = mask

        proxy_into = schema.Proxy()
        for i in range(len(self)):
            if mask is not None and not pmask[i]:
                continue
            yield schema.unpack_from(buf, index[i], idmap, proxy_class_new, proxy_into)

    def __len__(self):
        return len(self.index)

    @classmethod
    @cython.locals(schema = Schema, data_pos = cython.size_t, initial_pos = cython.size_t, current_pos = object,
        schema_pos = cython.size_t, schema_end = cython.size_t)
    def build(cls, initializer, destfile = None, tempdir = None, idmap = None,
            return_mapper = True, read_only = True):
        """
        Builds an array of objects with a uniform :class:`Schema` into a memory mapped temporary file.

        :param iterable initializer: Content of the array.

        :type destfile: file or file-like
        :param destfile: *(optional)* An explicit file where the mapping should be built. If ``return_mapper``
            is True (the default), this has to be an actual file. Otherwise, it can be any file-like object
            that supports seeking and overwriting. The array will be written at the current position,
            and mapped from it.

        :param str tempdir: *(optional)* A directory into which temporary files will be constructed. The build
            process needs temporary storage, so it will be used even when an explicit ``destfile`` is given.

        :type idmap: dict-like or StrongIdMap
        :param idmap: An :term:`idmap` to be used during the construction. If not given, a temporary
            :term:`idmap` is constructed for each object that is written, preventing instance deduplication
            across items but reducing memory usage.

        :param bool return_mapper: *(default True)* If false, only the final writing position will be returned,
            instead of the actual mapped array. This allows both embedding of the array into a larger
            structure (further objects can be appended at the returned position) and construction onto
            file-like objects (mapping is only supported from actual file objects, and not generally
            from file-like objects).

        :param bool read_only: *(optional)* Whether the mapping should be read-only, of if write access
            should also be requested. Defaults to true.

        :rtype: MappedArrayProxyBase or int
        :returns: Either the mapped array when ``return_mapper`` is True, or the position within the file
            where the array ends if it is False.
        """
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
            return cls.map_file(destfile, initial_pos, read_only = read_only)
        else:
            return final_pos

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        """
        Build a mapped array instance mapping the array in ``buf`` at position ``offset``

        :param buf: Readable buffer to map the array from

        :param int offset: *(optional)* Position within the buffer where the array is located.
        """
        return cls(buf, offset)

    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        Build a mapped array instance mapping the given ``fileobj`` at position ``offset``.
        A size can optionally be given to map only the necessary portion of the file.

        :param file fileobj: Memory-mappable file where the array is located

        :param int offset: *(optional)* Position within the file where the array is located.

        :param int size: *(optional)* Size of the array data. If given, it will be used to reduce
            the mapped portion of the file to the minimum necessary mapping.

        :param bool read_only: *(optional)* Whether the mapping should be read-only, or if
            write access should also be requested. Defaults to true.
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        fileobj.seek(offset)
        total_size = cls._Header.unpack(fileobj.read(cls._Header.size))[0]
        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY
        if read_only:
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE
        buf = mmap.mmap(fileobj.fileno(), total_size + offset - map_start,
            access = access, offset = map_start)
        rv = cls(buffer(buf, offset - map_start))
        rv._file = fileobj
        rv._mmap = buf
        if not read_only:
            offset -= map_start
            rv.wr_buf = (ctypes.c_char * (len(buf) - offset)).from_buffer(buf, offset)
        return rv


if cython.compiled:

    @cython.cfunc
    @cython.locals(
        hkey = numeric_A, elem = numeric_B,
        lo = cython.size_t, hi = cython.size_t, length = cython.size_t,
        mid = cython.size_t, mid2 = cython.size_t, stride0 = cython.size_t, hint = cython.size_t,
        pindex = cython.p_char, skip = cython.size_t, check_equal = cython.bint)
    @cython.returns(cython.size_t)
    def _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem):
        hi = length
        lo = 0

        if hkey < cython.cast('numeric_B *', pindex)[0]:
            if check_equal:
                return hi
            else:
                return lo
        elif hkey > cython.cast('numeric_B *', pindex + stride0 * (hi-1))[0]:
            return hi

        elem = cython.cast(cython.typeof(elem), hkey)
        if lo < hi:
            # First iteration a quick guess assuming uniform distribution of keys
            mid = min(hint, hi-1)
            mkey = cython.cast('numeric_B *', pindex + stride0 * mid)[0]
            if mkey < elem:
                # Got a lo guess, now skip-search forward for a hi
                lo = mid = mid+1
                skip = 32
                while skip > 0 and mid + skip < hi:
                    if cython.cast('numeric_B *', pindex + stride0 * (mid+skip))[0] < elem:
                        lo = mid+1
                        mid += skip
                        skip *= 2
                    else:
                        hi = mid + skip
                        break
            elif mkey > elem:
                # Got a hi guess, now skip-search backwards for a lo
                hi = mid
                skip = 32
                while skip > 0 and mid > lo + skip:
                    if cython.cast('numeric_B *', pindex + stride0 * (mid-skip))[0] > elem:
                        hi = mid
                        mid -= skip
                        skip *= 2
                    else:
                        mid -= skip
                        while mid > lo and cython.cast('numeric_B *', pindex + stride0 * (mid-1))[0] == elem:
                            mid -= 1
                        lo = mid
                        break
            else:
                # hit, but must find the first
                # good idea to go sequential because we assume collisions are unlikely
                while mid > lo and cython.cast('numeric_B *', pindex + stride0 * (mid-1))[0] == elem:
                    mid -= 1
                return mid
        # Final stretch: search the remaining range with a regular binary search
        while lo < hi:
            mid = (lo+hi)//2
            mkey = cython.cast('numeric_B *', pindex + stride0 * mid)[0]
            if mkey < elem:
                lo = mid+1
            elif mkey > elem:
                hi = mid
            else:
                while mid > lo and cython.cast('numeric_B *', pindex + stride0 * (mid-1))[0] == elem:
                    mid -= 1
                return mid
        # Check equality if requested
        if check_equal and lo < length and cython.cast('numeric_B *', pindex + stride0 * lo)[0] != elem:
            lo = length
        return lo

    @cython.cfunc
    @cython.locals(hkey = cython.ulonglong, elem = cython.ulonglong,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_ui64(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.slonglong, elem = cython.slonglong,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_i64(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.ulonglong, elem = cython.uint,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_ui32(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.slonglong, elem = cython.sint,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_i32(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.ulonglong, elem = cython.ushort,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_ui16(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.slonglong, elem = cython.sshort,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_i16(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.ulonglong, elem = cython.uchar,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_ui8(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.slonglong, elem = cython.schar,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_i8(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.double, elem = cython.double,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_f64(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

    @cython.cfunc
    @cython.locals(hkey = cython.float, elem = cython.float,
        pindex = cython.p_char, stride0 = cython.size_t, length = cython.size_t,
        hint = cython.size_t, check_equal = cython.bint)
    def _c_search_hkey_f32(hkey, pindex, stride0, length, hint, check_equal):
        elem = 0
        return _c_search_hkey_gen(hkey, pindex, stride0, length, hint, check_equal, elem)

if cython.compiled:
    # Commented cython directives in pxd
    #@cython.ccall
    @cython.locals(
        lo = cython.size_t, hi = cython.size_t, hint = cython.size_t, stride0 = cython.size_t,
        indexbuf = 'Py_buffer', pindex = cython.p_char, check_equal = cython.bint, ix = cython.size_t)
    #@cython.returns(cython.size_t)
    def _hinted_bsearch(a, hkey, hint, lo, hi, check_equal):
        """
        Does a binary search in "a" for "hkey", assuming the key is expected
        to be found around position "hint" or at most between "lo" and "hi".
        If check_equal is given and True, it will make sure to return exactly
        "hi" iff "hkey" is not found in "a". Otherwise, the position where it
        would be if it is there will be returned.
        """
        if hi <= lo:
            return lo

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
                ix = _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'I':
                ix = _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'l' or dtype == 'q':
                ix = _c_search_hkey_i64(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'i':
                ix = _c_search_hkey_i32(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'H':
                ix = _c_search_hkey_ui16(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'h':
                ix = _c_search_hkey_i16(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'B':
                ix = _c_search_hkey_ui8(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'b':
                ix = _c_search_hkey_i8(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'd':
                ix = _c_search_hkey_f64(hkey, pindex, stride0, hi, hint, check_equal)
            elif dtype == 'f':
                ix = _c_search_hkey_f32(hkey, pindex, stride0, hi, hint, check_equal)
            else:
                raise NotImplementedError("Unsupported array type %s" % (chr(dtype),))
        except OverflowError:
            # The conversion of hkey to either longlong or ulonglong failed
            # The key most certainly is not in the array, that can't hold the value at all,
            # but we must check (in python fashion to support the extended range numbers)
            # whether it lies on the left or right side
            if hkey < a[0]:
                ix = lo
            elif hkey > a[hi-1]:
                ix = hi
            else:
                raise
        finally:
            PyBuffer_Release(cython.address(indexbuf)) #lint:ok

        return ix

    def hinted_bsearch(a, hkey, hint):
        """
        Search into the sorted array ``a`` the value ``hkey``, assuming it should
        be close to ``hint``.

        :param ndarray a: Sorted array of one of the supported types (see :func:`bsearch`).

        :param hkey: Value to search for

        :param int hint: Expected location of ``hkey``

        :rtype: int
        :returns: Location where ``hkey`` should be, if it is there. See :func:`bsearch`.
        """
        hi = len(a)
        lo = 0
        return _hinted_bsearch(a, hkey, hint, lo, hi, False)
else:
    import bisect

    def _py__hinted_bsearch(a, hkey, hint, lo, hi, check_equal):
        ix = bisect.bisect_left(a, hkey)
        if check_equal and ix < hi and a[ix] != hkey:
            ix = hi
        return ix
    globals()['_hinted_bsearch'] = _py__hinted_bsearch

    def _py_hinted_bsearch(a, hkey, hint):
        """
        Search into the sorted array ``a`` the value ``hkey``, assuming it should
        be close to ``hint``.

        :param ndarray a: Sorted array of one of the supported types (see :func:`bsearch`).

        :param hkey: Value to search for

        :param int hint: Expected location of ``hkey``

        :rtype: int
        :returns: Location where ``hkey`` should be, if it is there. See :func:`bsearch`.
        """
        return bisect.bisect_left(a, hkey)
    _py_hinted_bsearch.__name__ = 'hinted_bsearch'
    globals()['hinted_bsearch'] = _py_hinted_bsearch

#@cython.ccall
@cython.locals(lo = cython.size_t, hi = cython.size_t)
#@cython.returns(cython.size_t)
def bsearch(a, hkey):
    """
    Search into the sorted array ``a`` the value ``hkey``.

    :param ndarray a: Sorted array of one of the supported types:

        * ints of 8, 16, 32 and 64 bits, signed and unsigned

        * floats of 32 and 64 bits

    :param hkey: Value to search for.

    :rtype: int
    :returns: Location where ``hkey`` should be, if it is there. The caller has to double-check.
        If the value isn't present, the returned index can be out of bounds. The return
        value has the same semantic as that of :func:`bisect.bisect_left`.
    """
    hi = len(a)
    lo = 0
    return _hinted_bsearch(a, hkey, (lo+hi)//2, lo, hi, False)

#@cython.ccall
@cython.locals(hi = cython.size_t, ix = cython.size_t, hint = cython.size_t)
#@cython.returns(cython.bint)
def hinted_sorted_contains(a, hkey, hint):
    """
    Search into the sorted array ``a`` the value ``hkey``, assuming it should be near ``hint``.

    See :func:`hinted_bsearch`.

    :rtype: bool
    :returns: Whether the value is present or not
    """
    hi = len(a)
    ix = _hinted_bsearch(a, hkey, hint, 0, hi, True)
    return ix < hi

#@cython.ccall
@cython.locals(hi = cython.size_t, ix = cython.size_t)
#@cython.returns(cython.bint)
def sorted_contains(a, hkey):
    """
    Search into the sorted array ``a`` the value ``hkey``, assuming it should be near ``hint``.

    See :func:`bsearch`.

    :rtype: bool
    :returns: Whether the value is present or not
    """
    hi = len(a)
    ix = _hinted_bsearch(a, hkey, hi//2, 0, hi, True)
    return ix < hi

if cython.compiled:

    @cython.nogil
    @cython.cfunc
    @cython.returns(cython.size_t)
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char,
        pend1 = cython.p_char, pend2 = cython.p_char, pdestend = cython.p_char, pdeststart = cython.p_char,
        ref = numeric_A)
    def _c_merge_gen(pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref):
        # Main merge
        pend1 = pindex1 + stride0 * length1
        pend2 = pindex2 + stride0 * length2
        pdestend = pdest + stride0 * destlength
        pdeststart = pdest
        while pindex1 < pend1 and pindex2 < pend2 and pdest < pdestend:
            ref = cython.cast('numeric_A *', pindex2)[0]
            while pindex1 < pend1 and cython.cast('numeric_A *', pindex1)[0] <= ref and pdest < pdestend:
                cython.cast('numeric_A *', pdest)[0] = cython.cast('numeric_A *', pindex1)[0]
                cython.cast('numeric_A *', pdest)[1] = cython.cast('numeric_A *', pindex1)[1]
                pdest += stride0
                pindex1 += stride0
            if pindex1 < pend1:
                ref = cython.cast('numeric_A *', pindex1)[0]
                while pindex2 < pend2 and cython.cast('numeric_A *', pindex2)[0] <= ref and pdest < pdestend:
                    cython.cast('numeric_A *', pdest)[0] = cython.cast('numeric_A *', pindex2)[0]
                    cython.cast('numeric_A *', pdest)[1] = cython.cast('numeric_A *', pindex2)[1]
                    pdest += stride0
                    pindex2 += stride0

        # Copy leftover tails
        while pindex1 < pend1 and pdest < pdestend:
            cython.cast('numeric_A *', pdest)[0] = cython.cast('numeric_A *', pindex1)[0]
            cython.cast('numeric_A *', pdest)[1] = cython.cast('numeric_A *', pindex1)[1]
            pdest += stride0
            pindex1 += stride0
        while pindex2 < pend2 and pdest < pdestend:
            cython.cast('numeric_A *', pdest)[0] = cython.cast('numeric_A *', pindex2)[0]
            cython.cast('numeric_A *', pdest)[1] = cython.cast('numeric_A *', pindex2)[1]
            pdest += stride0
            pindex2 += stride0
        return (pdest - pdeststart) // stride0

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.ulonglong,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.ulonglong](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.longlong,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.slonglong](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.uint,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.uint](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.sint](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui16(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.ushort](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i16(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.sshort](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_ui8(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.uchar](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.int,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_i8(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.schar](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.double,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_f64(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.double](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

    #@cython.cfunc
    @cython.locals(
        length1 = cython.size_t, length2 = cython.size_t, destlength = cython.size_t,
        stride0 = cython.size_t, ref = cython.float,
        pindex1 = cython.p_char, pindex2 = cython.p_char, pdest = cython.p_char)
    #@cython.returns(cython.size_t)
    def _c_merge_f32(pindex1, length1, pindex2, length2, pdest, destlength, stride0):
        ref = 0
        return _c_merge_gen[cython.float](
            pindex1, length1, pindex2, length2, pdest, destlength, stride0, ref)

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
                        with cython.nogil:
                            return _c_merge_ui64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'I':
                        with cython.nogil:
                            return _c_merge_ui32(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'l' or dtype == 'q':
                        with cython.nogil:
                            return _c_merge_i64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'i':
                        with cython.nogil:
                            return _c_merge_i32(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'H':
                        with cython.nogil:
                            return _c_merge_ui16(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'h':
                        with cython.nogil:
                            return _c_merge_i16(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'B':
                        with cython.nogil:
                            return _c_merge_ui8(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'b':
                        with cython.nogil:
                            return _c_merge_i8(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'd':
                        with cython.nogil:
                            return _c_merge_f64(pindex1, length1, pindex2, length2, pdest, destlength, stride0)
                    elif dtype == 'f':
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
        for i in range(0, len(parts), 2):
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

@cython.cfunc
@cython.locals(self = 'NumericIdMapper', elem = numeric_A, indexbuf = 'Py_buffer',
    pybuf = 'Py_buffer', startpos = int, hkey = cython.ulonglong,
    stride0 = cython.size_t, stride1 = cython.size_t,
    pindex = cython.p_char, pindexend = cython.p_char, nitems = int)
def _numeric_id_get_gen(self, elem, startpos, hkey, nitems):
    #lint:disable
    buf = self._likebuf
    PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
    try:
        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
        try:
            if (indexbuf.strides == cython.NULL
                   or indexbuf.ndim < 2
                   or indexbuf.len < nitems * indexbuf.strides[0]):
                raise ValueError("Invalid buffer state")
            stride0 = indexbuf.strides[0]
            stride1 = indexbuf.strides[1]
            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
            if pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
                return cython.cast('numeric_A *', pindex + stride1)[0]
        finally:
            PyBuffer_Release(cython.address(indexbuf))
    finally:
        PyBuffer_Release(cython.address(pybuf))
    #lint:enable

@cython.cclass
class NumericIdMapper(_CZipMapBase):
    """
    A numerical :term:`Id Mapper`, providing a mapping from 64-bit unsigned integers
    to 64-bit unsigned integers, adequate for object mappings where the keys are unsigned integers.

    As all :term:`Id Mapper`\ s, it implements a dict-like interface.
    """

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
        """
        Make sure the whole mapping is loaded into memory to prime it for faster future access.
        """
        # Just touch everything in sequential order
        self.index.max()

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
                        for i in range(self.index_elements):
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
                        for i in range(self.index_elements):
                            yield (
                                cython.cast(cython.p_uint, pindex)[0],
                                cython.cast(cython.p_uint, pindex + stride1)[0]
                            )
                            pindex += stride0
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in range(self.index_elements):
                        yield (
                            index[i,0],
                            index[i,1]
                        )
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in range(self.index_elements):
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
        if hi <= lo:
            return hi
        hikey = self._index_max
        lokey = self._index_min
        if hkey < lokey:
            return lo
        elif hkey > hikey:
            return hi
        if cython.compiled:
            dtype = self._dtype
            if dtype is npuint64 or dtype is npuint32 or dtype is npuint16 or dtype is npuint8:
                #lint:disable
                PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
                try:
                    if ( indexbuf.strides == cython.NULL
                            or indexbuf.len < hi * indexbuf.strides[0] ):
                        raise ValueError("Invalid buffer state")
                    pindex = cython.cast(cython.p_char, indexbuf.buf)
                    stride0 = indexbuf.strides[0]

                    if dtype is npuint64:
                        # Must be careful about overflow
                        if hikey > cython.cast(cython.ulonglong, 0xFFFFFFFFFFFF):
                            hint = lo + ((hkey - lokey) >> 32) * (hi - lo) // max(((hikey - lokey) >> 32), 1)
                        elif hikey > cython.cast(cython.ulonglong, 0xFFFFFFFF):
                            hint = lo + ((hkey - lokey) >> 16) * (hi - lo) // max(((hikey - lokey) >> 16), 1)
                        else:
                            hint = lo + (hkey - lokey) * (hi - lo) // max((hikey - lokey), 1)
                        return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint32:
                        hint = lo + (hkey - lokey) * (hi - lo) // max((hikey - lokey), 1)
                        return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint16:
                        hint = lo + (hkey - lokey) * (hi - lo) // max((hikey - lokey), 1)
                        return _c_search_hkey_ui16(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint8:
                        hint = lo + (hkey - lokey) * (hi - lo) // max((hikey - lokey), 1)
                        return _c_search_hkey_ui8(hkey, pindex, stride0, hi, hint, True)
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
        hkey = cython.ulonglong, startpos = int, nitems = int)
    def get(self, key, default = None):
        if not isinstance(key, (int, long)):
            return default
        if key < 0 or key > self.dtypemax:
            return default

        hkey = key
        nitems = self.index_elements
        if nitems == 0:
            return default

        startpos = self._search_hkey(hkey)
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                if dtype is npuint64:
                    return _numeric_id_get_gen[cython.ulonglong](self, 0, startpos, hkey, nitems)
                elif dtype is npuint32:
                    return _numeric_id_get_gen[cython.uint](self, 0, startpos, hkey, nitems)
                elif dtype is npuint16:
                    return _numeric_id_get_gen[cython.ushort](self, 0, startpos, hkey, nitems)
                elif dtype is npuint8:
                    return _numeric_id_get_gen[cython.uchar](self, 0, startpos, hkey, nitems)

            index = self.index
            if startpos < nitems and index[startpos,0] == hkey:
                return index[startpos,1]
        return default

    @classmethod
    @cython.locals(
        basepos = cython.Py_ssize_t, curpos = cython.Py_ssize_t, endpos = cython.Py_ssize_t, finalpos = cython.Py_ssize_t,
        discard_duplicates = cython.bint, discard_duplicate_keys = cython.bint)
    def build(cls, initializer, destfile = None, tempdir = None,
            discard_duplicates = False, discard_duplicate_keys = False, return_mapper = True, read_only = True):
        """
        Builds a :class:`NumericIdMapper` from the given iterable. The iterable should
        yield ``(key, value)`` pairs, in which both key and value are numbers fitting the range
        of the :term:`Id Mapper`.

        :param initializer: Iterable of items that will be built onto the resulting :term:`Id Mapper`.

        :type destfile: file or file-like
        :param destfile: *(optional)* An explicit file object where the mapper will be built. The same
            restrictions apply than in the case of :meth:`MappedArrayProxyBase.build`.

        :param str tempdir: Directory where temporary files may be placed if needed.

        :param bool discard_duplicates: If True, duplicate items will be removed from the mapping. Requires
            extra effort during build.

        :param bool discard_duplicate_keys: If True, duplicate keys will be discarded from the mapping. Only
            an arbitrary item will remain. Requires extra effort during build.

        :param bool return_mapper: Whether to return the mapped :term:`Id Mapper` or the ending offset.
            See the same argument of :meth:`MappedArrayProxyBase.build` for a detailed description.

        :param bool read_only: *(optional)* Whether the mapping should be read-only, or if
            write access should also be requested. Defaults to true.
        """
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
            write(b"\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos, read_only = read_only)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        """
        Build an :term:`Id Mapper` from the data in ``buf`` at position ``offset``

        :param buf: Readable buffer to map the :term:`Id Mapper` from

        :param int offset: *(optional)* Position within the buffer where the data is located.
        """
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'NumericIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        Build an :term:`Id Mapper` from the data in ``fileobj`` at position ``offset``.
        A size can optionally be given to map only the necessary portion of the file.

        :param file fileobj: Memory-mappable file where the data is located

        :param int offset: *(optional)* Position within the file where the data is located.

        :param int size: *(optional)* Size of the data. If given, it will be used to reduce
            the mapped portion of the file to the minimum necessary mapping.

        :param bool read_only: *(optional)* Whether the mapping should be read-only, or if
            write access should also be requested. Defaults to true.
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY

        if size is None:
            map_size = 0
        else:
            map_size = size + offset - map_start

        fileobj.seek(map_start)
        if read_only:
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE
        buf = mmap.mmap(fileobj.fileno(), map_size, access = access, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

    @classmethod
    @cython.locals(
        basepos = cython.Py_ssize_t, curpos = cython.Py_ssize_t, endpos = cython.Py_ssize_t, finalpos = cython.Py_ssize_t,
        discard_duplicates = cython.bint, discard_duplicate_keys = cython.bint, index_elements = cython.size_t,
        mapper = 'NumericIdMapper')
    def merge(cls, parts, destfile = None, tempdir = None,
            discard_duplicates = False, discard_duplicate_keys = False):
        """
        Merge two or more :class:`NumericIdMapper` s into a single one efficiently.

        :param iterable parts: Iterable of :class:`NumericIdMapper` instances to be merged.

        See :meth:`build` for a description of the remaining arguments.
        """
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
            write(b"\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        rv = cls.map_file(destfile, basepos, size = finalpos - basepos)
        destfile.seek(finalpos)
        return rv

@cython.cclass
class NumericId32Mapper(NumericIdMapper):
    """
    Like :class:`NumericIdMapper` but for 32-bit unsigned integer keys and values.
    Hence, more compact, but also more limited.
    """
    dtype = npuint32

@cython.ccall
@cython.locals(self = 'ObjectIdMapper', elem = numeric_A, indexbuf = 'Py_buffer',
    startpos = int, hkey = cython.ulonglong, stride0 = cython.size_t,
    stride1 = cython.size_t, pindex = cython.p_char, pindexend = cython.p_char)
def _obj_id_get_gen(self, elem, startpos, hkey, key, default):
    #lint:disable
    nitems = self.index_elements
    PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
    try:
        if (indexbuf.strides == cython.NULL
               or indexbuf.ndim < 2
               or indexbuf.len < nitems * indexbuf.strides[0]):
            raise ValueError("Invalid buffer state")
        stride0 = indexbuf.strides[0]
        stride1 = indexbuf.strides[1]
        pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
        pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
        while pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
            if self._compare_keys(self._buf,
                  cython.cast('numeric_A *', pindex + stride1)[0], key):
                return cython.cast('numeric_A *', pindex + 2*stride1)[0]
            pindex += stride0
        return default
    finally:
        PyBuffer_Release(cython.address(indexbuf))
    #lint:enable

@cython.cclass
class ObjectIdMapper(_CZipMapBase):
    """
    A generic :term:`Id Mapper`, providing a mapping from :term:`hashable objects`
    to 64-bit unsigned integers, adequate for object mappings where the keys are arbitrary :term:`hashable objects`.

    As all :term:`Id Mapper`\ s, it implements a dict-like interface.
    """

    dtype = npuint64

    # Num-items, Index-pos
    _Header = struct.Struct('=QQ')

    cython.declare(
        _buf = object,
        _likebuf = object,
        _file = object,
        _dtype = object,
        _basepos = cython.Py_ssize_t,
        index = object,
        index_elements = cython.ulonglong,
        index_offset = cython.ulonglong,
        _dtype_bits = cython.ushort,
    )

    @property
    def buf(self):
        return self._buf

    @property
    def offset(self):
        return self._basepos

    @property
    def fileobj(self):
        return self._file

    @property
    def fileno(self):
        return self._file.fileno()

    @property
    def name(self):
        return self._file.name

    @cython.locals(dtypemax = cython.ulonglong)
    def __init__(self, buf, offset = 0):
        # Accelerate class attributes
        self._dtype = self.dtype

        # Initialize buffer
        self._buf = buf
        self._likebuf = _likebuffer(buf)
        self._basepos = offset

        # Parse header and map index
        self.index_elements, self.index_offset = self._Header.unpack_from(self._buf, offset)

        self.index = numpy.ndarray(
            buffer = self._buf,
            offset = self.index_offset + offset,
            dtype = self.dtype,
            shape = (self.index_elements, 3))

        dtype = self.dtype
        try:
            dtypemax = int(~dtype(0))
        except:
            try:
                dtypemax = int(~dtype.type(0))
            except:
                dtypemax = ~0
        self._dtype_bits = 0
        while dtypemax:
            self._dtype_bits += 1
            dtypemax >>= 1

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
    @cython.locals(pos = cython.Py_ssize_t, sindex = cython.longlong, uindex = cython.ulonglong)
    def _unpack(self, buf, index):
        # None is represented with an offset of 1, which is an impossible offset
        # (it would point into the header, 0 would be the dict itself so it's valid)
        if index == 1:
            return None

        # Compute absolute offset out of relative index
        pos = self._basepos
        if not cython.compiled:
            sindex = int(index)
            if self._dtype_bits != 64 and (sindex & (1 << (self._dtype_bits-1))):
                # sign-extend
                sindex |= (~0) & ~((1 << (64 - self._dtype_bits)) - 1)

            if sindex & (1 << 63):
                # interpret two-s complement in python long arithmetic
                sindex = -(-sindex & 0xFFFFFFFFFFFFFFFF)
        else:
            # reinterpret-cast
            uindex = index
            sindex = uindex
            if self._dtype_bits != 64:
                # sign-extend
                sindex <<= 64 - self._dtype_bits
                sindex >>= 64 - self._dtype_bits
        return _mapped_object_unpack_from(buf, pos + sindex)

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer')
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
            if dtype is npuint64:
                PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_SIMPLE)
                try:
                    if indexbuf.len < (self.index_elements * stride * cython.sizeof(cython.ulonglong)):
                        raise ValueError("Invalid buffer state")
                    for i in range(self.index_elements):
                        yield self._unpack(self._buf,
                            cython.cast(cython.p_ulonglong, indexbuf.buf)[i*stride+offs])
                finally:
                    PyBuffer_Release(cython.address(indexbuf))
            elif dtype is npuint32:
                PyObject_GetBuffer(index, cython.address(indexbuf), PyBUF_SIMPLE)
                try:
                    if indexbuf.len < (self.index_elements * stride * cython.sizeof(cython.uint)):
                        raise ValueError("Invalid buffer state")
                    for i in range(self.index_elements):
                        yield self._unpack(self._buf,
                            cython.cast(cython.p_uint, indexbuf.buf)[i*stride+offs])
                finally:
                    PyBuffer_Release(cython.address(indexbuf))
            else:
                for i in range(self.index_elements):
                    if not make_sequential:
                        pos = index[i, offs]
                    else:
                        pos = index[i]
                    yield self._unpack(self._buf, pos)
            #lint:enable
        else:
            for i in range(self.index_elements):
                if not make_sequential:
                    pos = index[i, offs]
                else:
                    pos = index[i]
                yield self._unpack(buf, pos)

    def __iter__(self):
        return self.iterkeys()

    def itervalues(self):
        return iter(self.index[:,2])

    def values(self):
        return self.index[:,2]

    def keys(self):
        # Baaad idea
        return list(self.iterkeys())

    @cython.locals(i = cython.ulonglong, indexbuf = 'Py_buffer',
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
                    for i in range(self.index_elements):
                        yield (
                            self._unpack(buf,
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
                    for i in range(self.index_elements):
                        yield (
                            self._unpack(buf,
                                cython.cast(cython.p_uint, pindex + stride1)[0]),
                            cython.cast(cython.p_uint, pindex + 2*stride1)[0]
                        )
                        pindex += stride0
                finally:
                    PyBuffer_Release(cython.address(indexbuf))
            else:
                for i in range(self.index_elements):
                    yield (
                        self._unpack(self._buf, index[i,1]),
                        index[i,2]
                    )
            #lint:enable
        else:
            for i in range(self.index_elements):
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
        if hi <= lo:
            return hi
        if cython.compiled:
            dtype = self._dtype
            if dtype is npuint64 or dtype is npuint32 or dtype is npuint16 or dtype is npuint8:
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
                        return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint32:
                        # A quick guess assuming uniform distribution of keys over the 64-bit value range
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint16:
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui16(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint8:
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui8(hkey, pindex, stride0, hi, hint, True)
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
        hkey = cython.ulonglong, startpos = int, nitems = int)
    def get(self, key, default = None):
        nitems = self.index_elements
        if nitems == 0:
            return default

        hkey = _stable_hash(key)
        startpos = self._search_hkey(hkey)
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                buf = self._likebuf
                if dtype is npuint64:
                    return _obj_id_get_gen[cython.ulonglong](
                        self, 0, startpos, hkey, key, default)
                elif dtype is npuint32:
                    return _obj_id_get_gen[cython.uint](
                        self, 0, startpos, hkey, key, default)
                elif dtype is npuint16:
                    return _obj_id_get_gen[cython.ushort](
                        self, 0, startpos, hkey, key, default)
                elif dtype is npuint8:
                    return _obj_id_get_gen[cython.uchar](
                        self, 0, startpos, hkey, key, default)

            index = self.index
            while startpos < nitems and index[startpos,0] == hkey:
                if self._compare_keys(buf, index[startpos, 1], key):
                    return index[startpos,2]
                startpos += 1
        return default

    @classmethod
    @cython.locals(
        basepos = cython.Py_ssize_t, curpos = cython.Py_ssize_t, endpos = cython.Py_ssize_t, finalpos = cython.Py_ssize_t,
        dtypemax = cython.ulonglong, implicit_offs = cython.Py_ssize_t, widmap = StrongIdMap, kpos = cython.longlong,
        ukpos = cython.ulonglong)
    def build(cls, initializer, destfile = None, tempdir = None, return_mapper = True,
            min_buf_size = 128, idmap = None, implicit_offs = 0, read_only = True):
        """
        Builds the mapper from the iterable of items passed in ``initializer``. The items should be
        ``(key, value)`` pairs where keys can be any :term:`hashable object`, and values are integers
        in the supported range for this mapper.

        See :meth:`NumericIdMapper.build` for the other arguments.
        """
        if destfile is None:
            destfile = tempfile.NamedTemporaryFile(dir = tempdir)

        dtype = cls.dtype
        try:
            dtypemax = int(~dtype(0))
        except:
            try:
                dtypemax = int(~dtype.type(0))
            except:
                dtypemax = ~0

        basepos = destfile.tell()

        # Reserve space for the header
        write = destfile.write
        write(cls._Header.pack(0, 0))

        if idmap is None:
            idmap = widmap = StrongIdMap()
        elif isinstance(idmap, StrongIdMap):
            widmap = idmap
        else:
            widmap = None

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
        curpos = cls._Header.size
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

                # None will be represented with an offset of 1, which is an impossible offset
                # (it would point into the header, 0 would be the dict itself so it's valid)
                if k is None:
                    part.append((_stable_hash(k), 1, i))
                    continue

                # these are wrapped objects, not plain objects, so make sure they have distinct xid
                kid = wrapped_id(k)
                if kid in idmap:
                    kpos = idmap[kid]
                    kpos -= (basepos + implicit_offs)
                    if kpos > dtypemax:
                        raise ValueError("Cannot represent offset with requested precision")

                    # Must cast into unsigned, since idmapper dtypes are always unsigned,
                    # but object relative offsets might be negative
                    ukpos = kpos
                    if not cython.compiled:
                        ukpos &= dtypemax

                    part.append((_stable_hash(k), ukpos, i))
                else:
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
                            endpos = pack_into(k, valbuf, 0, idmap, curpos + basepos + implicit_offs)
                            break
                        except (struct.error, IndexError):
                            klen += klen

                    if valbuflen < endpos:
                        raise RuntimeError("Buffer overflow")
                    write(valbuf[:endpos])

                    idmap[kid] = curpos + basepos + implicit_offs
                    if widmap is not None:
                        widmap.link(kid, k)

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
            write(b"\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos))
        destfile.seek(finalpos)
        destfile.flush()

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos, read_only = read_only)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        """
        See :meth:`NumericIdMapper.map_buffer`
        """
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'ObjectIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        See :meth:`NumericIdMapper.map_file`
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY

        if size is None:
            map_size = 0
        else:
            map_size = size + offset - map_start

        fileobj.seek(map_start)
        if read_only:
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE
        buf = mmap.mmap(fileobj.fileno(), map_size, access = access, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

def safe_utf8(x):
    if isinstance(x, six.text_type):
        # The assignment-style cast is needed because an inline case triggers a Cython compiler crash
        ux = x
        return ux.encode("utf8")
    else:
        return x

@cython.ccall
@cython.locals(self = 'StringIdMapper', elem = numeric_A, pbkey = 'const char *',
    blen = cython.size_t, indexbuf = 'Py_buffer', startpos = int, hkey = cython.ulonglong,
    stride0 = cython.size_t, stride1 = cython.size_t, pbuf_ptr = 'const char *')
def _str_id_get_gen(self, elem, pbkey, blen, startpos, hkey, pbuf_ptr, pbuf_len, default):
    nitems = self.index_elements
    PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
    try:
        if (indexbuf.strides == cython.NULL
               or indexbuf.ndim < 2
               or indexbuf.len < nitems * indexbuf.strides[0]):
            raise ValueError("Invalid buffer state")
        stride0 = indexbuf.strides[0]
        stride1 = indexbuf.strides[1]
        pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
        pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
        while pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
            elem = cython.cast('numeric_A *', pindex + stride1)[0]
            if _compare_bytes_from_cbuffer(pbkey, blen, pbuf_ptr,
                    cython.cast(cython.size_t, elem), pbuf_len):
                return cython.cast('numeric_A *', pindex + 2*stride1)[0]
            pindex += stride0
        return default
    finally:
        PyBuffer_Release(cython.address(indexbuf))

@cython.cclass
class StringIdMapper(_CZipMapBase):
    """
    An :term:`Id Mapper`, providing a mapping from strings
    to 64-bit unsigned integers, adequate for object mappings where the keys are unsigned integers.

    As all :term:`Id Mapper`\ s, it implements a dict-like interface.
    """

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
                        for i in range(self.index_elements):
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
                        for i in range(self.index_elements):
                            yield _unpack_bytes_from_cbuffer(
                                cython.cast(cython.p_char, pybuf.buf),
                                cython.cast(cython.p_uint, indexbuf.buf)[i*stride+offs],
                                pybuf.len, None)
                    finally:
                        PyBuffer_Release(cython.address(indexbuf))
                else:
                    for i in range(self.index_elements):
                        yield _unpack_bytes_from_cbuffer(
                            cython.cast(cython.p_char, pybuf.buf),
                            index[i],
                            pybuf.len, None)
            finally:
                PyBuffer_Release(cython.address(pybuf))
            #lint:enable
        else:
            for i in range(self.index_elements):
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
                        for i in range(self.index_elements):
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
                        for i in range(self.index_elements):
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
                    for i in range(self.index_elements):
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
            for i in range(self.index_elements):
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
        if hi <= lo:
            return hi
        if cython.compiled:
            dtype = self._dtype
            if dtype is npuint64 or dtype is npuint32 or dtype is npuint16 or dtype is npuint8:
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
                        return _c_search_hkey_ui64(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint32:
                        # A quick guess assuming uniform distribution of keys over the 64-bit value range
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui32(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint16:
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui16(hkey, pindex, stride0, hi, hint, True)
                    elif dtype is npuint8:
                        hint = ((hkey * (hi-lo)) >> 32) + lo
                        return _c_search_hkey_ui8(hkey, pindex, stride0, hi, hint, True)
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
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes, stride0 = cython.size_t,
        stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *', pybuf = 'Py_buffer')
    def get(self, key, default = None):
        if not isinstance(key, basestring):
            return default

        nitems = self.index_elements
        if nitems == 0:
            return default

        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()
        startpos = self._search_hkey(hkey)
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
                        return _str_id_get_gen[cython.ulonglong](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint32:
                        return _str_id_get_gen[cython.uint](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint16:
                        return _str_id_get_gen[cython.ushort](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint8:
                        return _str_id_get_gen[cython.uchar](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(p_char, pybuf.buf), pybuf.len, default)
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
        basepos = cython.Py_ssize_t, curpos = cython.Py_ssize_t, endpos = cython.Py_ssize_t, finalpos = cython.Py_ssize_t,
        dtypemax = cython.ulonglong)
    def build(cls, initializer, destfile = None, tempdir = None, return_mapper = True, read_only = True):
        """
        Builds the mapper from the iterable of items passed in ``initializer``. The items should be
        ``(key, value)`` pairs where keys are strings, and the values are integers
        in the supported range for this mapper.

        See :meth:`NumericIdMapper.build` for the other arguments.
        """
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
            write(b"\x00" * (32 - (finalpos & 31)))
            finalpos = destfile.tell()

        destfile.seek(basepos)
        write(cls._Header.pack(nitems, indexpos - basepos))
        destfile.seek(finalpos)
        destfile.flush()

        if return_mapper:
            rv = cls.map_file(destfile, basepos, size = finalpos - basepos, read_only = read_only)
            destfile.seek(finalpos)
        else:
            rv = finalpos
        return rv

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        """
        See :meth:`NumericIdMapper.map_buffer`
        """
        return cls(buf, offset)

    @classmethod
    @cython.locals(rv = 'StringIdMapper')
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        See :meth:`NumericIdMapper.map_file`
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        map_start = offset - offset % mmap.ALLOCATIONGRANULARITY

        if size is None:
            map_size = 0
        else:
            map_size = size + offset - map_start

        fileobj.seek(map_start)
        if read_only:
            access = mmap.ACCESS_READ
        else:
            access = mmap.ACCESS_WRITE
        buf = mmap.mmap(fileobj.fileno(), map_size, access = access, offset = map_start)
        rv = cls(buf, offset - map_start)
        rv._file = fileobj
        return rv

@cython.cclass
class StringId32Mapper(StringIdMapper):
    """
    An :term:`Id Mapper` like :class:`StringIdMapper` whose values are 32-bit unsigned integers,
    and thus more compact, but limited to smaller indexes.
    """
    dtype = npuint32
    xxh = xxhash.xxh32

@cython.ccall
@cython.locals(self = 'NumericIdMapper', elem = numeric_A, rv = list, hkey = cython.ulonglong,
    startpos = int, pybuf = 'Py_buffer', indexbuf = 'Py_buffer',
    pindex = cython.p_char, pindexend = cython.p_char)
def _numeric_id_multi_get_gen(self, elem, rv, hkey, startpos, default):
    #lint:disable
    nitems = self.index_elements
    PyObject_GetBuffer(self._likebuf, cython.address(pybuf), PyBUF_SIMPLE)
    try:
        try:
            PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
            if (indexbuf.strides == cython.NULL
                   or indexbuf.ndim < 2
                   or indexbuf.len < nitems * indexbuf.strides[0]):
                raise ValueError("Invalid buffer state")
            stride0 = indexbuf.strides[0]
            stride1 = indexbuf.strides[1]
            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
            while pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
                rv.append(cython.cast('numeric_A *', pindex + stride1)[0])
                pindex += stride0

            return rv if rv else default
        finally:
            PyBuffer_Release(cython.address(indexbuf))
    finally:
       PyBuffer_Release(cython.address(pybuf))
    #lint:enable

@cython.ccall
@cython.locals(self = 'NumericIdMultiMapper', elem = numeric_A, startpos = int,
    hkey = cython.ulonglong, nitems = int, pybuf = 'Py_buffer',
    indexbuf = 'Py_buffer', pindex = cython.p_char, pindexend = cython.p_char)
def _numeric_id_multi_has_gen(self, elem, startpos, hkey):
    #lint:disable
    nitems = self.index_elements
    PyObject_GetBuffer(self._likebuf, cython.address(pybuf), PyBUF_SIMPLE)
    try:
        PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
        try:
            if (indexbuf.strides == cython.NULL
                   or indexbuf.ndim < 2
                   or indexbuf.len < nitems * indexbuf.strides[0]):
                raise ValueError("Invalid buffer state")
            stride0 = indexbuf.strides[0]
            pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
            pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
            return pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey
        finally:
            PyBuffer_Release(cython.address(indexbuf))
    finally:
        PyBuffer_Release(cython.address(pybuf))
    #lint:enable

@cython.cclass
class NumericIdMultiMapper(NumericIdMapper):
    """
    A numeric :term:`Id Multi Mapper`, providing a mapping from 64-bit unsigned integers
    to 64-bit unsigned integers, adequate for object mappings where the keys are unsigned integers.

    As all :term:`Id Multi Mapper`\ s, it implements a dict-like interface whose values are lists
    of matches, rather than singular matches.
    """

    @cython.ccall
    def _encode_key(self, key):
        return key

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int)
    def get(self, key, default = None):
        key = self._encode_key(key)
        if not isinstance(key, (int, long)):
            return default
        try:
            hkey = key
        except OverflowError:
            if key < 0 or key > self.dtypemax:
                return default

        nitems = self.index_elements
        if nitems == 0:
            return default

        startpos = self._search_hkey(hkey)
        rv = []
        if 0 <= startpos < nitems:
            dtype = self._dtype
            if cython.compiled:
                if dtype is npuint64:
                    return _numeric_id_multi_get_gen[cython.ulonglong](
                        self, 0, rv, hkey, startpos, default)
                elif dtype is npuint32:
                    return _numeric_id_multi_get_gen[cython.uint](
                        self, 0, rv, hkey, startpos, default)
                elif dtype is npuint16:
                    return _numeric_id_multi_get_gen[cython.ushort](
                        self, 0, rv, hkey, startpos, default)
                elif dtype is npuint8:
                    return _numeric_id_multi_get_gen[cython.uchar](
                        self, 0, rv, hkey, startpos, default)

            index = self.index
            while startpos < nitems and index[startpos,0] == hkey:
                rv.append(index[startpos,1])
                startpos += 1
            if rv:
                return rv
        return default

    def __contains__(self, key):
        return self.has_key(key)

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int)
    def has_key(self, key):
        key = self._encode_key(key)
        if not isinstance(key, (int, long)):
            return False
        if key < 0 or key > self.dtypemax:
            return False

        hkey = key
        nitems = self.index_elements
        if nitems == 0:
            return False

        startpos = self._search_hkey(hkey)
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                if dtype is npuint64:
                    return _numeric_id_multi_has_gen[cython.ulonglong](
                        self, 0, startpos, hkey)
                elif dtype is npuint32:
                    return _numeric_id_multi_has_gen[cython.uint](
                        self, 0, startpos, hkey)
                elif dtype is npuint16:
                    return _numeric_id_multi_has_gen[cython.ushort](
                        self, 0, startpos, hkey)
                elif dtype is npuint8:
                    return _numeric_id_multi_has_gen[cython.uchar](
                        self, 0, startpos, hkey)

            index = self.index
            return startpos < nitems and index[startpos,0] == hkey

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int,
        stride0 = cython.size_t, stride1 = cython.size_t,
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def get_iter(self, key):
        """
        Like :meth:`~NumericIdMapper.get`, except it returns an iterator instead of an actual list.
        It can be faster when there are a large number of matches.
        """
        key = self._encode_key(key)
        if not isinstance(key, (int, long)):
            return
        if key < 0 or key > self.dtypemax:
            return

        hkey = key
        nitems = self.index_elements
        if nitems == 0:
            return

        startpos = self._search_hkey(hkey)
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

@cython.cclass
class NumericId32MultiMapper(NumericIdMultiMapper):
    """
    A numeric :term:`Id Multi Mapper`, providing a mapping from 32-bit unsigned integers
    to 32-bit unsigned integers, adequate for object mappings where the keys are unsigned integers.

    Like :class:`NumericIdMultiMapper`, but more compact and limited to smaller indexes.
    """
    dtype = npuint32

@cython.ccall
@cython.locals(self = 'StringIdMultiMapper', elem = numeric_A, rv = list, pbkey = 'const char *',
    blen = cython.size_t, startpos = int, hkey = cython.ulonglong, pbuf_ptr = 'const char *',
    pbuf_len = cython.size_t, indexbuf = 'Py_buffer', stride0 = cython.size_t,
    stride1 = cython.size_t, pindex = cython.p_char)
def _str_id_multi_get_gen(self, elem, rv, pbkey, blen, startpos, hkey, pbuf_ptr, pbuf_len, default):
    nitems = self.index_elements
    PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
    try:
        if (indexbuf.strides == cython.NULL
               or indexbuf.ndim < 2
               or indexbuf.len < nitems * indexbuf.strides[0]):
            raise ValueError("Invalid buffer state")
        stride0 = indexbuf.strides[0]
        stride1 = indexbuf.strides[1]
        pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
        pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
        while pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
            elem = cython.cast('numeric_A *', pindex + stride1)[0]
            if _compare_bytes_from_cbuffer(pbkey, blen, pbuf_ptr,
                    cython.cast(cython.size_t, elem), pbuf_len):
                rv.append(cython.cast('numeric_A *', pindex + 2*stride1)[0])
            pindex += stride0
        return rv if rv else default
    finally:
        PyBuffer_Release(cython.address(indexbuf))

@cython.ccall
@cython.locals(self = 'StringIdMultiMapper', elem = numeric_A, pbkey = 'const char *',
    startpos = int, pbuf_ptr = 'const char *', pbuf_len = cython.size_t,
    hkey = cython.ulonglong, blen = cython.size_t, indexbuf = 'Py_buffer', stride0 = cython.size_t,
    stride1 = cython.size_t, pindex = cython.p_char, pindexend = cython.p_char)
def _str_id_multi_has_gen(self, elem, pbkey, blen, startpos, hkey, pbuf_ptr, pbuf_len):
    #lint:disable
    nitems = self.index_elements
    PyObject_GetBuffer(self.index, cython.address(indexbuf), PyBUF_STRIDED_RO)
    try:
        if (indexbuf.strides == cython.NULL
               or indexbuf.ndim < 2
               or indexbuf.len < nitems * indexbuf.strides[0]):
            raise ValueError("Invalid buffer state")
        stride0 = indexbuf.strides[0]
        stride1 = indexbuf.strides[1]
        pindex = cython.cast(cython.p_char, indexbuf.buf) + startpos * stride0
        pindexend = cython.cast(cython.p_char, indexbuf.buf) + indexbuf.len - stride0 + 1
        while pindex < pindexend and cython.cast('numeric_A *', pindex)[0] == hkey:
            elem = cython.cast('numeric_A *', pindex + stride1)[0]
            if _compare_bytes_from_cbuffer(pbkey, blen, pbuf_ptr,
                    cython.cast(cython.size_t, elem), pbuf_len):
                return True
            pindex += stride0
        return False
    finally:
        PyBuffer_Release(cython.address(indexbuf))
    #lint:enable

@cython.cclass
class StringIdMultiMapper(StringIdMapper):
    """
    An :term:`Id Multi Mapper`, providing a mapping from strings
    to 64-bit unsigned integers, adequate for object mappings where the keys are strings.

    As all :term:`Id Multi Mapper`\ s, it implements a dict-like interface whose values are lists
    of matches, rather than singular matches.
    """

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        blen = cython.size_t, pbkey = 'const char *', pybuf = 'Py_buffer')
    def get(self, key, default = None):
        if not isinstance(key, basestring):
            return default
        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()

        nitems = self.index_elements
        if nitems == 0:
            return default

        startpos = self._search_hkey(hkey)
        if 0 <= startpos < nitems:
            buf = self._buf
            rv = []
            dtype = self._dtype
            if cython.compiled:
                pbkey = bkey
                blen = len(bkey)
                #lint:disable
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    if dtype is npuint64:
                        return _str_id_multi_get_gen[cython.ulonglong](
                            self, 0, rv, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint32:
                        return _str_id_multi_get_gen[cython.uint](
                            self, 0, rv, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint16:
                        return _str_id_multi_get_gen[cython.ushort](
                            self, 0, rv, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len, default)
                    elif dtype is npuint8:
                        return _str_id_multi_get_gen[cython.uchar](
                            self, 0, rv, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len, default)
                    else:
                        index = self.index
                        while startpos < nitems and index[startpos,0] == hkey:
                            if _compare_bytes_from_cbuffer(pbkey, blen,
                                    cython.cast(cython.p_char, pybuf.buf),
                                    self.index[startpos,1],
                                    pybuf.len) == bkey:
                                rv.append(index[startpos,2])
                            startpos += 1
                finally:
                    PyBuffer_Release(cython.address(pybuf))
                #lint:enable
            else:
                index = self.index
                while startpos < nitems and index[startpos,0] == hkey:
                    if _unpack_bytes_from_pybuffer(buf, index[startpos,1], None) == bkey:
                        rv.append(index[startpos,2])
                    startpos += 1
                if rv:
                    return rv
        return default

    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        blen = cython.size_t, pbkey = 'const char *', pybuf = 'Py_buffer', indexbuf = 'Py_buffer',
        stride0 = cython.size_t, stride1 = cython.size_t, pindex = cython.p_char)
    def get_iter(self, key):
        """
        See :meth:`NuericIdMultiMapper.get_iter`
        """
        if not isinstance(key, basestring):
            return

        nitems = self.index_elements
        if nitems == 0:
            return

        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()
        startpos = self._search_hkey(hkey)
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

    def __contains__(self, key):
        return self.has_key(key)

    @cython.ccall
    @cython.locals(
        hkey = cython.ulonglong, startpos = int, nitems = int, bkey = bytes,
        stride0 = cython.size_t, stride1 = cython.size_t, blen = cython.size_t, pbkey = 'const char *',
        indexbuf = 'Py_buffer', pybuf = 'Py_buffer', pindex = cython.p_char)
    def has_key(self, key):
        if not isinstance(key, basestring):
            return False

        nitems = self.index_elements
        if nitems == 0:
            return False

        bkey = self._encode(key)
        hkey = self._xxh(bkey).intdigest()
        startpos = self._search_hkey(hkey)
        if 0 <= startpos < nitems:
            buf = self._buf
            dtype = self._dtype
            if cython.compiled:
                pbkey = bkey
                blen = len(bkey)
                buf = self._likebuf
                PyObject_GetBuffer(buf, cython.address(pybuf), PyBUF_SIMPLE)
                try:
                    if dtype is npuint64:
                        return _str_id_multi_has_gen[cython.ulonglong](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len)
                    elif dtype is npuint32:
                        return _str_id_multi_has_gen[cython.uint](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len)
                    elif dtype is npuint16:
                        return _str_id_multi_has_gen[cython.ushort](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len)
                    elif dtype is npuint8:
                        return _str_id_multi_has_gen[cython.uchar](
                            self, 0, pbkey, blen, startpos, hkey,
                            cython.cast(cython.p_char, pybuf.buf), pybuf.len)
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

@cython.cclass
class StringId32MultiMapper(StringIdMultiMapper):
    """
    An :term:`Id Multi Mapper` like :class:`StringIdMultiMapper`, where values are limited
    to 32-bit unsigned integers, and thus more compact but limited.
    """
    dtype = npuint32
    xxh = xxhash.xxh32

@cython.cclass
class ApproxStringIdMultiMapper(NumericIdMultiMapper):
    """
    An :term:`Approximate Id Multi Mapper`, providing a mapping from strings
    to 64-bit unsigned integers, adequate for object mappings where the keys are strings
    and a small number of false matches are acceptable. No mapping is done
    if the provided key is already an integer.

    As all :term:`Approximate Id Multi Mapper`\ s, it implements a dict-like interface whose values are lists
    of matches, rather than singular matches.
    """
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
    def _encode_key(self, key):
        if isinstance(key, (int, long)):
            return key
        else:
            return self._xxh(self._encode(key)).intdigest()

    @classmethod
    def build(cls, initializer, *p, **kw):
        """
        Constructs the mapping from an iterable of items. The items should be ``(key, value)`` pairs
        where keys are strings, and values are integers that fit the range of the mapper.

        See :meth:`NumericIdMultiMapper.build` for the other arguments.
        """
        xxh = cls.xxh
        encode = cls.encode
        def wrapped_initializer():
            for key, value in initializer:
                if isinstance(key, (int, long)):
                    yield key, value
                else:
                    yield xxh(encode(key)).intdigest(), value
        return super(ApproxStringIdMultiMapper, cls).build(wrapped_initializer(), *p, **kw)

@cython.cclass
class ApproxStringId32MultiMapper(ApproxStringIdMultiMapper):
    """
    A numeric :term:`Approximate Id Multi Mapper`, providing a mapping from strings
    to 32-bit unsigned integers, adequate for object mappings where the keys are unsigned integers.

    Like :class:`ApproxStringIdMultiMapper`, but more compact and limited to smaller indexes,
    with a false positive rate slightly higher as well.
    """
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
            i += 1
            value_cache[value] = i
            dump((key, i), keys_file, 2)
        else:
            dump((key, value_cache[value]), keys_file, 2)
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
    """
    Base class for mappings of keys to objects with a uniform :class:`Schema`.

    Construct a concrete class by subclassing and providing an array class and an :term:`Id Mapper`::

        class SomeArrayType(MappedArrayProxyBase):
            schema = Schema.from_typed_slots(SomeClass)

        class SomeMappingType(MappedMappingProxyBase):
            ValueArray = SomeArrayType
            IdMapper = NumericIdMapper

    Then build them into temporary files by using :meth:`build`::

        mapped_mapping = SomeMappingType.build(iterable)

    The returned mapping will be memory-mapped from a temporary file. You can also provide
    an explicit file where to build the mapping instead. See :meth:`build` for details.

    The schema is pickled into the buffer so the mapping should be portable.

    The class implements a (readonly) dict-like interface, supporting iteration of keys, values and items,
    length, and random access subscripting.

    Class attributes:

    .. attribute:: IdMapper

        An implementation of an :term:`Id Mapper` that will provide the key-position mapping. The mapper
        should be wide enough to contain references into the :attr:`ValueArray`.

    .. attribute:: ValueArray

        A concrete subclass of :class:`MappedArrayProxyBase` that will provide the schema for values and
        will be indexed by the :term:`Id Mapper`.
    """

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
        """
        Builds a mapping of keys to objects with a uniform :class:`Schema` into a memory mapped temporary file.

        :param iterable initializer: Content of the mapping.

        :param file destfile: *(optional)* An explicit file where the mapping should be built.
            This has to be an actual file, since it needs to be memory-mapped.
            The mapping will be written at the current position, and memory-mapped from it.

        :param str tempdir: *(optional)* A directory into which temporary files will be constructed. The build
            process needs temporary storage, so it will be used even when an explicit ``destfile`` is given.

        :type idmap: dict-like or StrongIdMap
        :param idmap: An :term:`idmap` to be used during the construction. If not given, a temporary
            :term:`idmap` is constructed for each object that is written, preventing instance deduplication
            across items but reducing memory usage.

        :param dict value_array_kwargs: Custom keyword arguments to be passed when invoking
            :attr:`ValueArray` . :meth:`~MappedArrayProxyBase.build`.

        :param dict id_mapper_kwargs: Custom keyword arguments to be passed when invoking
            :attr:`IdMapper` . :meth:`~NumericIdMapper.build`.

        :rtype: MappedMappingProxyBase
        :returns: The constructed mapping
        """
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
                destfile.write(b"\x00" * (32 - (pos & 31)))

            values_pos = destfile.tell()

            blocklen = 1 << 20
            for start in range(0, len(value_array.buf), blocklen):
                destfile.write(buffer(value_array.buf, start, blocklen))
            destfile.write(cls._Footer.pack(values_pos - initial_pos))
            destfile.flush()

            return cls(value_array, id_mapper)

    @classmethod
    def map_buffer(cls, buf, offset = 0):
        """
        Builds a mapping proxy out of the data in ``buf`` at offset ``offset``.

        The way mappings are constructed requires metadata to be at a footer, and not a header.
        This means the buffer should end where the mapping ends, or the mapping won't be
        read correctly. If the mapping is embedded on a larger buffer, a slice must be taken
        prior to calling this method, so the caller needs to know the size of the mapping beforehand.

        :param buffer buf: A read buffer where the data is located. The mapping must end where the buffer ends.

        :param int offset: *(optional)* The offset where the mapping starts.
        """
        values_pos, = cls._Footer.unpack_from(buf, offset + len(buf) - cls._Footer.size)
        value_array = cls.ValueArray.map_buffer(buf, offset + values_pos)
        id_mapper = cls.IdMapper.map_buffer(buf, offset)
        return cls(value_array, id_mapper)

    @classmethod
    def map_file(cls, fileobj, offset = 0, size = None, read_only = True):
        """
        Builds a mapping proxy out of the data in ``fileobj`` at offset ``offset``
        and size ``size``.

        The way mappings are constructed requires metadata to be at a footer, and not a header.
        This means the buffer should end where the mapping ends, or the mapping won't be
        read correctly. If the mapping is embedded on a larger buffer, a slice must be taken
        prior to calling this method, so the caller needs to know the size of the mapping beforehand.

        :param file fileobj: A file object where the data is located. The mapping must end where the file ends,
            or an explicit ``size`` must be given.

        :param int offset: *(optional)* The offset where the mapping starts.

        :param int size: *(optional)* The size of the mapping relative to its starting offset. It must be
            given if the mapping doesn't end at the EOF, to be able to locate the footer.

        :param bool read_only: *(optional)* Whether the mapping should be read-only, or if
            write access should also be requested. Defaults to true.
        """
        if isinstance(fileobj, zipfile.ZipExtFile):
            return cls.map_zipfile(fileobj, offset, size, read_only = read_only)

        # If no size is given, it's the whole file by default
        if size is None:
            fileobj.seek(0, os.SEEK_END)
            size = fileobj.tell() - offset

        # Read the footer
        fileobj.seek(offset + size - cls._Footer.size)
        values_pos, = cls._Footer.unpack(fileobj.read(cls._Footer.size))
        fileobj.seek(offset)

        # Map everything
        id_mapper = cls.IdMapper.map_file(fileobj, offset, size = values_pos, read_only = read_only)
        value_array = cls.ValueArray.map_file(fileobj, offset + values_pos,
            size = size - cls._Footer.size - values_pos, read_only = read_only)
        return cls(value_array, id_mapper)


class MappedMultiMappingProxyBase(MappedMappingProxyBase):
    """
    A base class for mappings like :class:`MappedMappingProxyBase`, but which accepts multiple
    values for a key. It has to be paired with :term:`Id Multi Mapper` s instead.
    """
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
