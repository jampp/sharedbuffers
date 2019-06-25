.. _decimal-internals:

Decimal
=======

Decimal numbers are stored as an arbitrary-precision decimal floating-point. The mantissa (digits) is stored
as a :ref:`sequence <sequence-internals>` of integers from 0 to 9. That means it will take one byte per decimal digit,
plus the sequence header.

.. code-block:: C

    struct Decimal {
        long long exponent_sign;
        List digits;
    };

.. attribute:: exponent_sign

    Both the exponent and sign, computed as::

        exponent_sign = (exponent << 1) | sign

.. attribute:: digits

    The mantissa as a :ref:`sequence <sequence-internals>` of integers from 0 to 9.

The exponent, sign and digits match the semantics of :meth:`decimal.Decimal.as_tuple`.

Examples
--------

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> num = Decimal('38.872874')
    >>> end = mapped_decimal.pack_into(num, buf, 0)
    >>> buf[:end]
    bytearray(b'\xf4\xff\xff\xff\xff\xff\xff\xffB\x08\x00\x00\x03\x08\x08\x07\x02\x08\x07\x04\x00\x00\x00\x00')
    >>> struct.unpack_from('=q', buf[:end])
    (-12,)
    >>> num.as_tuple()
    DecimalTuple(sign=0, digits=(3, 8, 8, 7, 2, 8, 7, 4), exponent=-6)

    >>> num = Decimal('-38.872874')
    >>> end = mapped_decimal.pack_into(num, buf, 0)
    >>> buf[:end]
    bytearray(b'\xf5\xff\xff\xff\xff\xff\xff\xffB\x08\x00\x00\x03\x08\x08\x07\x02\x08\x07\x04\x00\x00\x00\x00')
    >>> struct.unpack_from('=q', buf[:end])
    (-11,)
    >>> num.as_tuple()
    DecimalTuple(sign=1, digits=(3, 8, 8, 7, 2, 8, 7, 4), exponent=-6)

