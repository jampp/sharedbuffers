from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITABLE, PyBUF_STRIDED_RO, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy, memcmp

cdef packed struct _varstr_header:
    unsigned short shortlen
    unsigned long long biglen
