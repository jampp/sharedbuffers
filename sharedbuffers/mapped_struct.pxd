from cpython.buffer cimport PyBUF_SIMPLE, PyBUF_WRITABLE, PyBUF_STRIDED_RO, PyObject_GetBuffer, PyBuffer_Release
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GT, Py_GE
from cpython.bytes cimport PyBytes_FromStringAndSize
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

cdef inline int popcount(unsigned long long x):
    return __builtin_popcountll(x)

cdef inline void mfence_full():
    __sync_synchronize()

# NOTE: This is not the same as the fused type 'numeric', because CAS
# doesn't work with floating point types.
ctypedef fused atomic_type:
    char
    unsigned char
    short
    unsigned short
    int
    unsigned int
    long
    unsigned long
    long long
    unsigned long long

cdef inline bint _c_atomic_cas(atomic_type *ptr, atomic_type exp_val, atomic_type new_val):
    return __sync_bool_compare_and_swap(ptr, exp_val, new_val)

# These functions will probably trigger all sorts of warnings related to
# aliasing issues. Good thing Python is compiled with -fno-strict-aliasing ;)

cdef inline bint _c_atomic_cas_flt(float *ptr, float exp_val, float new_val):
    cdef int *exp_ptr = <int *>&exp_val
    cdef int *new_ptr = <int *>&new_val
    return __sync_bool_compare_and_swap(<int *>ptr, exp_ptr[0], new_ptr[0])

cdef inline bint _c_atomic_cas_dbl(double *ptr, double exp_val, double new_val):
    cdef long long *exp_ptr = <long long *>&exp_val
    cdef long long *new_ptr = <long long *>&new_val
    return __sync_bool_compare_and_swap(<long long *>ptr, exp_ptr[0], new_ptr[0])
