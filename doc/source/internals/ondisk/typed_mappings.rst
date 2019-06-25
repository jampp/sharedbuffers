.. _typed-mapping-internals:

Typed mappings
==============

Typed mappings are similar in their interface to :ref:`idmapper-internals`, but support complex objects
as values, rather than just primitive integer types.

Internally, they are a composition of an :ref:`id mapper <idmapper-internals>` and a
:ref:`typed array <array-internals>`.

They differ from general :ref:`mapping-internals` in that they're specialized for large scale structures
that are much bigger than RAM, and are thus constructed from item pair generators rather than from native dicts.

They are constructed by subclassing :class:`~sharedbuffers.mapped_struct.MappedMappingProxyBase` to
specify concrete implementations for the :attr:`~sharedbuffers.mapped_struct.MappedMappingProxyBase.IdMapper`
and :attr:`~sharedbuffers.mapped_struct.MappedMappingProxyBase.ValueArray`,
and invoking `MappedMappingProxyBase.build` on that subclass.

Binary format
-------------

Instead of a header, typed mappings have a footer, so they're located by their end, and not their start.

This makes them easier to build in a streaming fashion, but it makes embedding them into larger structures
a bit harder, so they're usually embedded within zip files instead. Since uncompressed entries within zipfiles
can be mapped just as if they were a regular file, this is a practical way to combine multiple typed mappings into a
more complex "bundle".

The typed mapping structure is thus:

.. code-block: C

    struct TypedMappingFooter {
        unsigned long long values_pos;
    };

    struct TypedMapping {
        IndexType index;
        char padding[];
        ArrayType values;
        TypedMappingFooter footer;
    }

Since :ref:`array-internals` have the schema embedded through :ref:`schema-serialization`, these are highly
portable across schema changes.
