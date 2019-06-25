.. _schema-serialization:

Schema serialization
====================

`Schemas <Schema>` are currently serialized by pickling them.

Schemas implement a ``__getstate__`` that exports only the relevant state, which includes
attribute types, order and alignment settings:

.. c:type:: SchemaState

    .. attribute:: slot_types

        A dictionary from attribute name to attribute type. Attribute types here refer to packer
        instances (like `mapped_object`). So :class:`list` would be replaced by :class:`mapped_list`, and
        so on.

        When an attribute's type references another known `Schema`, it will be referenced and pickled
        in here inside an instance of `mapped_object_with_schema`.

    .. attribute:: slot_keys

        A tuple with all the attributes, in the order they'll appear in-buffer.

    .. attribute:: bases

        A tuple with base classes to be applied to the automatically generated proxies, if any,
        or None if generic proxies are to be used.

    .. attribute:: alignment

        The alignment size used throughout the `Schema`.

For example::

    {
        'slot_types': {
            'bignum': <class 'sharedbuffers.mapped_struct.int64'>,
            'fraction': <class 'sharedbuffers.mapped_struct.float32'>,
            'bigger_fraction': <class 'sharedbuffers.mapped_struct.float64'>,
            'yetanother': <class 'sharedbuffers.mapped_struct.mapped_object'>,
            'otherstruct': <sharedbuffers.mapped_struct.mapped_object_with_schema object at 0x7faef62c2b50>,
            'smallnum': <class 'sharedbuffers.mapped_struct.byte'>
        },
        'slot_keys': (
            'bigger_fraction', 'bignum', 'otherstruct', 'yetanother', 'fraction', 'smallnum'
        ),
        'bases': None,
        'alignment': 8
    }
