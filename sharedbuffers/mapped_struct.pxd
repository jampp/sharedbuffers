from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITABLE, PyBUF_STRIDED_RO, PyObject_GetBuffer, PyBuffer_Release
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GT, Py_GE
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.array cimport array
from libc.string cimport memcpy, memcmp
from libc.math cimport isinf, isnan

cdef packed struct _varstr_header:
    unsigned short shortlen
    unsigned long long biglen

cpdef size_t _hinted_bsearch(a, hkey, size_t hint, size_t lo, size_t hi, bint check_equal) except 0xFFFFFFFFFFFFFFFF
cpdef size_t hinted_bsearch(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef size_t bsearch(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef bint hinted_sorted_contains(a, hkey, size_t hint) except 0xFFFFFFFFFFFFFFFF
cpdef bint sorted_contains(a, hkey) except 0xFFFFFFFFFFFFFFFF
cpdef bint proxied_list_richcmp(a, b, char op) except 0xFFFFFFFFFFFFFFFF
cpdef int proxied_list_cmp(a, b) except 0x02
cpdef bint proxied_list_eq(a, b) except 0xFFFFFFFFFFFFFFFF

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
cdef size_t _c_merge_ui16(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_i16(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_ui8(
    char* pindex1, size_t length1, char* pindex2, size_t length2,
    char* pdest, size_t destlength, size_t stride0) nogil except 0xFFFFFFFFFFFFFFFF
cdef size_t _c_merge_i8(
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
    cdef void __sync_synchronize()
    cdef bint __sync_bool_compare_and_swap(void *ptr, ...)
    cdef void __sync_add_and_fetch(void *ptr, ...)

cdef inline int popcount(unsigned long long x):
    return __builtin_popcountll(x)

cdef inline void mfence_full():
    __sync_synchronize()

# NOTE: This is not the same as the fused type 'numeric', because CAS
# doesn't work with floating point types.
ctypedef fused atomic_type:
    signed char
    unsigned char
    signed short
    unsigned short
    signed int
    unsigned int
    signed long
    unsigned long
    signed long long
    unsigned long long

cdef inline bint _c_atomic_cas(atomic_type *ptr, atomic_type exp_val, atomic_type new_val):
    return __sync_bool_compare_and_swap(ptr, exp_val, new_val)

cdef inline void _c_atomic_add(atomic_type *ptr, atomic_type value):
    __sync_add_and_fetch(ptr, value)

# These functions will probably trigger all sorts of warnings related to
# aliasing issues. Good thing Python is compiled with -fno-strict-aliasing ;)

cdef inline bint _c_atomic_cas_flt(float *ptr, float exp_val, float new_val):
    return __sync_bool_compare_and_swap(<int *>ptr,
        (<int *>&exp_val)[0], (<int *>&new_val)[0])

cdef inline bint _c_atomic_cas_dbl(double *ptr, double exp_val, double new_val):
    return __sync_bool_compare_and_swap(<long long *>ptr,
        (<long long *>&exp_val)[0], (<long long *>&new_val)[0])

cdef inline void _c_atomic_add_flt(float *ptr, float value):
    cdef float tmp
    while 1:
        tmp = ptr[0]
        if _c_atomic_cas_flt(ptr, tmp, tmp + value):
            break

cdef inline void _c_atomic_add_dbl(double *ptr, double value):
    cdef double tmp
    while 1:
        tmp = ptr[0]
        if _c_atomic_cas_dbl(ptr, tmp, tmp + value):
            break
