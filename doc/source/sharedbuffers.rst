sharedbuffers package
=====================

sharedbuffers.mapped\_struct module
-----------------------------------

.. automodule:: sharedbuffers.mapped_struct
    :member-order: groupwise
    :members: MappedArrayProxyBase, GenericFileMapper,
        NumericIdMapper, NumericId32Mapper, NumericIdMultiMapper, NumericId32MultiMapper,
        ObjectIdMapper, StringIdMapper, StringId32Mapper, StringIdMultiMapper, StringId32MultiMapper,
        ApproxStringIdMultiMapper, ApproxStringId32MultiMapper,
        MappedMappingProxyBase, MappedMultiMappingProxyBase,
        bsearch, hinted_bsearch, sorted_contains, hinted_sorted_contains, index_merge
    :undoc-members:
    :show-inheritance:

    For a high-level overview of how to use this module,
    check out :ref:`using-sharedbuffers`.

    .. autoclass:: Schema
        :members:
        :show-inheritance:
        :member-order: groupwise
        :no-undoc-members:

        See :ref:`using-sharedbuffers` for examples.

    .. autoclass:: mapped_object
        :members: pack_into, unpack_from, register_schema, register_subclass
        :show-inheritance:
        :undoc-members:

        Check out :ref:`composite-types` for examples on how to use this class.

mapped\_struct module internals
...............................

These are public functions and classes that, while useful, are meant for advanced use.
Make sure you're using the higher level concepts properly before resorting to these
lower-level primitives.

.. autoclass:: BufferProxyObject
    :show-inheritance:
    :members: _reset, _init
    :undoc-members:

.. autoclass:: StrongIdMap
    :show-inheritance:
    :undoc-members:

.. autoclass:: mapped_object_with_schema
    :undoc-members:

.. autofunction:: GenericProxyClass

sharedbuffers.pool module
-------------------------

.. automodule:: sharedbuffers.pool
    :member-order: groupwise
    :members: BaseObjectPool, TemporaryObjectPool
    :undoc-members:
    :show-inheritance:
