.. _array-internals:

Typed arrays
============

Typed arrays are a higher-level data structure that allows construction of large-scale
shared buffers consisting of arrays of objects sharing a particular `Schema`. Thus, they're uniformly shaped,
and the schema can be left implicit, requiring no :term:`RTTI` (compared to :ref:`sequence-internals`).

They're not usable as attribute slot types within `Schema` s, but they can be combined with other
high-level structures to create even more complex ones, like :ref:`typed-mapping-internals`.

Typed arrays are built with subclasses of :class:`~sharedbuffers.mapped_struct.MappedArrayProxyBase`.

A particular distinction with other :ref:`sequences <sequence-internals>` is that typed arrays apply
:ref:`schema-serialization` to make them more portable and self-described.

The format of typed arrays has evolved, and thus is one of the few that has been versioned.
Version-0, the initial version, contains no version tag, but it can be detected by verifying
some conditions that are only met for that particular version. Later versions include a version tag
in the header, and thus readers can check that they properly support the format before moving forward.

.. code-block:: C

    struct TypedArrayHeaderV0 {
        uint64_t total_size;
        uint64_t index_offset;
        uint64_t index_elements;
    };

    struct TypedArrayHeaderV1 {
        TypedArrayHeaderV0 v0;
        uint64_t version;
        uint64_t min_reader_version;
        uint64_t schema_offset;
        uint64_t schema_size;
    };

    union TypedArrayHeader {
        TypedArrayHeaderV0 v0;
        TypedArrayHeaderV1 v1;
    };

    union ArrayIndex {
        uint32_t ui32_positions[];
        uint64_t ui64_positions[];
    };

    struct TypedArray {
        TypedArrayHeader header;
        ArrayIndex index;
        char data[];
    };

Version-0 arrays can be detected because the first position in the index for these will always be less than 56,
which would place the object inside the header for Version-1 and above.

.. attribute:: total_size

    This is the aggregate size of the whole structure. If a parser needs to skip the whole array,
    it can skip that many bytes from the start of the header. Useful to be able to concatenate multiple high-level
    structures as building blocks of an even higher-level data structure.

.. attribute:: index_offset

    Offset relative to the header start where the index is located.

.. attribute:: index_elements

    The number of elements in the index, and thus indirectly the size of the typed array.

.. attribute:: version

    The version of this data structure.

.. attribute:: min_reader_version

    The minimum version the reader must be able to understand to have a chance to correctly
    parse this data structure. This provides some forward compatibility. If new optional features are added to the
    format, this number will be kept at an earlier version to allow old readers to still be able to parse the
    structure (albeit without the new, optional features).

.. attribute:: schema_offset

    The position, relative to the header start, of the serialized schema data. The contents of which
    are defined in the :ref:`schema-serialization` section.

.. attribute:: schema_size

    The size of serialized schema data.

.. attribute:: index

    The index is an array of :attr:`index_elements` 32 or 64-bit unsigned relative pointers.
    While most other data structures use signed relative pointers, these typed arrays use unsigned pointers,
    and they should always point within the structure and not go beyond the bounds defined in the structure's headers.
    These limitations make them more easily relocatable, which is useful when they're used as building blocks.
