Glossary
========

.. glossary::

    RTTI
        Runtime type information. A wrapper around an object of varying type that specifies the type
        using pre-registered typecodes, so readers can properly interpret the value at runtime.

    idmap
        A dict-like object used to deduplicate references to the same object, either during packing or
        unpacking. See :ref:`idmap usage <idmap-usage>` for proper usage of idmaps when building large collections.

    Id Mapper
        An implementation of a shared index, a mappable dict-like object specialized for mapping some kind
        of key into integer indexes or offsets. Used as a building block to build most of the dict-like types.
        See :ref:`container-structures` for usage examples.

    Id Multi Mapper
        Like :term:`Id Mapper`, but where multiple values for a key are acceptable. Their interface is also
        of a dict-like object, but where values are lists of matches instead.

    Approximate Id Multi Mapper
        Like :term:`Id Multi Mapper`, but where the association between keys and values may be approximate,
        and "extra" values might be returned with low probability. Pretty much like valued bloom filters.

    hashable object
    hashable objects
        As in python, mappings require that objects be hashable. In contrast with python, sharedbuffers needs
        a stable hash implementation (one that will be portable across processes and implementations). As such,
        not every hashable python object is hashable for sharedbuffers' purposes. See :py:func:`_stable_hash` for
        details.