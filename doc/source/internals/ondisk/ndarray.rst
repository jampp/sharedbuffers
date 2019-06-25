.. _ndarray-internals:

Numpy arrays
============

Numpy arrays are stored as a header, pointing to a dtype, and a :ref:`buffer <buffer-internals>`. Multi-dimensional
arrays also include a shape tuple just after the header:

.. code-block:: C

    struct NDArrayHeader {
        unsigned long long dtype_offs;
        unsigned long long data_offs;
    }

    struct DType {
        RTTIHeader rtti;
        char data[];
        char padding[];
    };

    struct NDArrayND {
        NDArrayHeader header;
        List shape;
        DType dtype;
        BufferData buffer;
    };

    struct NDArray1D {
        NDArrayHeader header;
        DType dtype;
        BufferData buffer;
    };

    union NDArray {
        NDArray1D arr_1d;
        NDArrayND arr_nd;
    };

.. c:type:: NDArrayHeader

    .. attribute:: dtype_offs

        The pointer, relative to the ``NDArrayHeader`` location, where the :attr:`dtype` may be found.

    .. attribute:: data_offs

        The pointer, relative to the ``NDArrayHeader`` location, where the :attr:`buffer` may be found.

.. c:type:: NDArrayND

    .. attribute:: shape

        A tuple describing the matrix shape. One-dimensional arrays and scalars don't need an explicit shape,
        since it can be implied by the buffer length, and so this attribute is missing. It is possible to guess
        which kind of array is being read by checking whether :attr:`dtype_offs` points just after the header's end.

.. c:type:: NDArray

    .. attribute:: dtype

        A :term:`RTTI` wrapped value that describes the array's data type. See :ref:`dtype-internals`
        and :ref:`rtti-wrapping`.

    .. attribute:: buffer

        A :ref:`byte buffer <buffer-internals>` holding the array's data.

.. _dtype-internals:

DTypes
------

Data types can be stored in one of 2 ways.

The most common data types are stored as a number, an id into the following table:

====  =============  ==========
id    dtype          typecodes
====  =============  ==========
0     uint64         L, I8
1     int64          l, i8
2     uint32         I, I4
3     int32          i, i4
4     uint16         H, I2
5     int16          h, i2
6     uint8          B, I1
7     int8           b, i1
8     float64        d, f8
9     float32        f, f4
N/A   bool8          ?, b1
N/A   complex64      F
N/A   complex128     D
N/A   complex256     G
====  =============  ==========

Typecodes can be prepended with an endianness mark, ``<`` for little endian, ``>`` for big endian,
and ``|`` for "doesn't matter" (single-byte types). For more on dtype strings,
check :ref:`numpy's documentation <arrays.dtypes>`.

Other data types are stored as a single object (which may itself be a sequence or a string), which is given to
`numpy.dtype` to construct a dtype instance.

Simple data types are described by a type code string. In contrast, :term:`structured data type` s are described by
a sequence of ``(name, dtype)`` pairs, in which each ``dtype`` in itself can be a string or yet another sequence
describing a nested structure on its own.

Examples
--------

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = proxied_ndarray.pack_into(numpy.arange(10), buf, 0)
    >>> buf[:end]
    bytearray(
        b'\x10\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00\x00\x00'
        b'q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'P\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
        b'\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'
        b'\x05\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00'
        b'\x07\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00'
        b'\t\x00\x00\x00\x00\x00\x00\x00')

    >>> end = proxied_ndarray.pack_into(numpy.arange(10).astype(numpy.int8), buf, 0)
    >>> buf[:end]
    bytearray(
        b'\x10\x00\x00\x00\x00\x00\x00\x00 \x00\x00\x00\x00\x00\x00\x00'
        b'q\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\n\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t')

Both of the above show a 1-d array of a simple data type.

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> buf[:end]
    bytearray(
        b'\x18\x00\x00\x00\x00\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x00'
        b'B\x02\x00\x00\x03\x03\x00\x00'
        b'q\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x12\x00\x00\x00\x00\x00\x00\x00'
        b'\x01\x00\x02\x00\x03\x00\x05\x00\x04\x00\x03\x00\xff\xff\xfe\xff\x03\x00')

The above example shows a 2-d matrix with an explicit shape tuple and the flattened buffer at the end.

.. code-block:: pycon

    >>> end = proxied_ndarray.pack_into(
        numpy.array(
            [(1,2,3,True), (5,4,3,False), (-1, -2 ,3,True)],
            numpy.dtype([
                ('f1', numpy.int32),
                ('f2', numpy.int8),
                ('f3', numpy.uint8),
                ('bv', numpy.bool8)
            ])),
            buf, 0)
    >>> buf[:end]
    bytearray(
        b'\x10\x00\x00\x00\x00\x00\x00\x00\t\x01\x00\x00\x00\x00\x00\x00'
        b'e\x02\x00\x00\x03\x03\x00\x00'
            # dtype index
            b'T\x04\x00\x00\x00\x00\x00\x00'
            b'\x18\x00\x00\x00I\x00\x00\x00'
            b'\x81\x00\x00\x00\xb9\x00\x00\x00'

            # dtype[0]
            b't\x00\x02\x00\x03\x00\x05\x00'
            b'T\x02\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x1c\x00\x00\x00'
                b's\x00\x00\x00\x00\x00\x00\x00\x02\x00f1'  # dtype[0][0]
                b's\x00\x00\x00\x00\x00\x00\x00\x03\x00<i4'  # dtype[0][1]

            # dtype[1]
            b't\x00\x00\x00\x00\x00\x00\x00'
            b'T\x02\x00\x00\x00\x00\x00\x00\x17\x00\x00\x00#\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00'
                b's\x00\x00\x00\x00\x00\x00\x00\x02\x00f2'  # dtype[1][0]
                b's\x00\x00\x00\x00\x00\x00\x00\x03\x00|i1'  # dtype[1][1]
            b't\x00\x00\x00\x00\x00\x00\x00'

            # dtype[2]
            b'T\x02\x00\x00\x00\x00\x00\x00\x17\x00\x00\x00#\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00'
                b's\x00\x00\x00\x00\x00\x00\x00\x02\x00f3'  # dtype[2][0]
                b's\x00\x00\x00\x00\x00\x00\x00\x03\x00|u1'  # dtype[2][1]

            # dtype[3]
            b't\x00\x00\x00\x00\x00\x00\x00'
            b'T\x02\x00\x00\x00\x00\x00\x00\x17\x00\x00\x00#\x00\x00\x00'
            b'\x00\x00\x00\x00\x00\x00\x00'
                b's\x00\x00\x00\x00\x00\x00\x00\x02\x00bv'  # dtype[3][0]
                b's\x00\x00\x00\x00\x00\x00\x00\x03\x00|b1'  # dtype[3][1]

            # buffer
            b'\x15\x00\x00\x00\x00\x00\x00\x00'
            b'\x01\x00\x00\x00\x02\x03\x01'
            b'\x05\x00\x00\x00\x04\x03\x00'
            b'\xff\xff\xff\xff\xfe\x03\x01')

This last example is a bit more interesting, showing a structured type with varying field data types. As can be seen,
field names are included in the dtype data, and buffer data is packed tightly (no alignment added).
