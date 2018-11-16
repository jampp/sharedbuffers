from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITABLE, PyBUF_STRIDED_RO, PyObject_GetBuffer, PyBuffer_Release
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GT, Py_GE
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.string cimport memcpy, memcmp

cdef packed struct _varstr_header:
    unsigned short shortlen
    unsigned long long biglen

cpdef size_t hinted_bsearch(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef size_t bsearch(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef bint hinted_sorted_contains(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef bint sorted_contains(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef bint proxied_list_richcmp(a, b, char op) except 0xFFFFFFFFFFFFFFFF
cpdef int proxied_list_cmp(a, b) except 0x02

cdef size_t _c_merge_ui64(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_i64(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_ui32(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_i32(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_f64(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_f32(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF

cpdef size_t index_merge(index1, index2, dest) except 0xFFFFFFFFFFFFFFFF
cpdef unsigned long long _stable_hash(key) except? 0

cdef extern from *:
    cdef int __builtin_popcountll(unsigned long long x)

cdef inline int popcount(unsigned long long x):
    return __builtin_popcountll(x)
