sharedbuffers package
=====================

sharedbuffers.mapped\_struct module
-----------------------------------

.. automodule:: sharedbuffers.mapped_struct
    :members: Schema, MappedArrayProxyBase, GenericFileMapper,
        NumericIdMapper, NumericId32Mapper, NumericIdMultiMapper, NumericId32MultiMapper,
        ObjectIdMapper,
        StringIdMapper, StringId32Mapper, StringIdMultiMapper, StringId32MultiMapper,
        ApproxStringIdMultiMapper, ApproxStringId32MultiMapper,
        MappedMappingProxyBase, MappedMultiMappingProxyBase, StrongIdMap,
        bsearch, hinted_bsearch, sorted_contains, hinted_sorted_contains, index_merge
    :undoc-members:
    :show-inheritance:

    For a high-level overview of how to use this module,
    check out :ref:`using-sharedbuffers`.

.. autoclass:: mapped_object
    :members: pack_into, unpack_from, register_schema, register_subclass
    :undoc-members:

    Check out :ref:`composite-types` for examples on how to use this class.

sharedbuffers.pool module
-------------------------

.. automodule:: sharedbuffers.pool
    :members: BaseObjectPool, TemporaryObjectPool
    :undoc-members:
    :show-inheritance:

