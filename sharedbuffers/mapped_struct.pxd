from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITABLE, PyBUF_STRIDED_RO, PyObject_GetBuffer, PyBuffer_Release
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy, memcmp

cdef packed struct _varstr_header:
    unsigned short shortlen
    unsigned long long biglen

cpdef size_t hinted_bsearch(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef size_t bsearch(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef bint hinted_sorted_contains(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef bint sorted_contains(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef size_t index_merge(index1, index2, dest) except 0xFFFFFFFFFFFFFFFF
