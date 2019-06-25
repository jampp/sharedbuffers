.. _mapping-internals:

Mapping types
=============

.. _dict-internals:

dict
----

Dicts are very similar to :ref:`typed-mapping-internals`, except they employ generic, dynamically-typed
id mappers and value sequences. Values and keys in dicts are :term:`RTTI` wrapped, so they support arbitrary
and heterogeneous keys and values, as long as keys are hashable by :func:`sharedbuffers.mapped_struct._stable_hash`.

Also, they don't do :ref:`schema-serialization`, so they depend on an implicit schema for any custom objects
that may be referred.

.. code-block:: C

    struct DictHeader {
        unsigned long long values_pos;
    };

    struct Dict {
        DictHeader header;
        ExactIndex index;
        List values;
    };

.. c:type:: DictHeader

    .. attribute:: values_pos

        The position, relative to the header start, where the values list is located.

.. c:type:: Dict

    .. attribute:: index

        The key-value :ref:`id mapper <idmapper-internals>`. This will always be an instance of
        :class:`~sharedbuffers.mapped_struct.ObjectIdMapper`, where values contain indices into
        the :attr:`values` list.

    .. attribute:: values

        The values associated with the keys, in no particular order. Values attached to the keys by the :attr:`index`
        are positions within this list.

Examples
--------

An empty dict, it can be rather fat.

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = proxied_dict.pack_into({}, buf, 0)
    >>> buf[:end]
    bytearray(
        b'(\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'B\x00\x00\x00\x00\x00\x00\x00')

Even a simple dict can be heavy:

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = proxied_dict.pack_into({'a': 35, 102: 'one-oh-two', ('a', 35): ('a', 'one-oh-two')}, buf, 0)
    >>> buf[:end]
    bytearray(
        # dict header
        b'\xa8\x00\x00\x00\x00\x00\x00\x00'

        # index header
        b'\x03\x00\x00\x00\x00\x00\x00\x00S\x00\x00\x00\x00\x00\x00\x00'

        # keys
        b's\x00\x00\x00\x00\x00\x00\x00\x01\x00a'
        b't\x00\x00\x00\x00\x00\x00\x00'
            b'T\x02\x00\x00\x00\x00\x00\x00\xed\xff\xff\xff\x10\x00\x00\x00'
        b'q#\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'qf\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00'

        # index
        b'f\x00\x00\x00\x00\x00\x00\x00C\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00'
        b'C\x1b\x0e\xb5\xcb\xd0\x7f\x02\x1b\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
        b'[n\x8c\xa9\xf1\xc4N\xd2\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

        # padding
        b'\x00\x00\x00\x00\x00'

        # values header
        b'T\x03\x00\x00\x00\x00\x00\x00'

        # values index
        b'\x18\x00\x00\x00(\x00\x00\x00K\x00\x00\x00'
        b'\x00\x00\x00\x00'

        # values data
        b'q#\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b't\x00\x00\x00\x00\x00\x00\x00'
            b'T\x02\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x1b\x00\x00\x00'
        b's\x00\x00\x00\x00\x00\x00\x00\x01\x00a'
        b's\x00\x00\x00\x00\x00\x00\x00\n\x00one-oh-two')

Still, querying this dict is quite fast, thanks to all the indexes that help locate values quickly even in large dicts.
