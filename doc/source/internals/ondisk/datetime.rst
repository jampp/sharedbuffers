.. _datetime-internals:

Datetime and Date
=================

Both date and datetime are stored as a unix timestamp. Datetimes are stored with microsecond precision,
and dates merely with second precision.

.. code-block:: C

    struct datetime {
        long long ts;
    };

    typedef datetime date;

For datetimes, the timestamp is computed according to the formula::

    ts = (int(timestamp) << 20) + frac(timestamp) * 1000000

Examples
--------

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_datetime.pack_into(datetime.datetime(2019, 03, 05, 11, 15, 24, 388001), buf, 0)
    >>> buf[:end]
    bytearray(b'\xa1\xeb\xc5G\xe8\xc7\x05\x00')
    >>> struct.unpack_from('=q', buf[:end])
    (1627175334046625,)

    >>> end = mapped_date.pack_into(datetime.date(2019, 03, 05), buf, 0)
    >>> buf[:end]
    bytearray(b'0\xe6}\\\x00\x00\x00\x00')
    >>> struct.unpack_from('=q', buf[:end])
    (1551754800,)
