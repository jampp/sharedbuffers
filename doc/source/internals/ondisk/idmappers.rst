.. _idmapper-internals:

ID mappers
==========

:term:`ID mapper` s are typed mappings used for indexing into large scale data structures. ID mappers map from a key
space of some uniform type, to unsigned indexes into a "target" data structure. Usually :ref:`array-internals`.

There are implementations for numerical and string keys. There are also :term:`Id Multi Mapper` s, that can return
more than a single match per key, and :term:`Approximate Id Multi Mapper` s, that are allowed to return false
positive pairings, in exchange for a significantly more compact representation.

Compact hash indexes
--------------------

Most ID mappers are designed around compact hash indexes.

A compact hash index is an array of ``(hash, key, value)`` entries, sorted by hash.

They're similar to a hash table in that computing a hash can provide a pretty good guess of the location within
the table for a particular key, if present. Like a hash table, that guess can be wrong, and a probe may be required
to find out for sure whether the key is present in the table or not.

They're different in how the probe is performed. Regular hash tables use probing algorithms with very poor access
patterns for secondary memory, and very poor worst-case performance (``O(N)``). Sharedbuffer's compact hash indexes
use an `exponential search`_ instead, that has an access pattern that's quite benign when the buffers are being
swapped from secondary memory. Also, compact hash indexes don't have empty slots. All entries are useful,
hence why they're "compact".

In essence, compact hash indexes are just ordered lists. Efficient to build in a batch operation, any sorting
algorithm will do, and efficient for querying with any algorithm tailored for sorted sequences. Our algorithm
is just a refinement of binary search, exploiting a few assumptions/expectations:

* First, the fact that the array, being filled with hashes, will contain a uniform key distribution. A good guess
  of an expected location within the array can thus be made, and the actual location of the key being searched
  shouldn't be far.
* Second, these indexes will be memory-mapped from secondary memory. This means locality of reference is far more
  important than counting comparison operations, so we try to minimize how many memory pages we'll touch and in which
  order when finding the accurate location of the key. So after checking the direction of the error (too high or
  too low), an exponential search in the appropriate direction finishes the job with minimal swapping overhead, if
  there is a need to swap pages.

Those optimizations make the data structure efficient for read-only operation from disk, even for very large
data sets, several times the size of the available RAM. When the data set fits in RAM, the operation is still as
efficient as a binary search at least. On average, it's actually faster, since it's ``O(log E)`` with ``E`` being the
error in predicted locations, which tends to be much smaller than ``N`` (the size of the index).

They're simpler and more compact than B-trees (no interior nodes) or MPH tables (no indirect lookup tables), although
both would be fine alternatives with different tradeoffs, that might be implemented at some future time.

Binary format
-------------

Compact hash indexes come in 2 major variations:

* Explicit-key entries for exact lookups
* Implicit-key entries for approximate lookups

They can also use various item "widths" (64-bit or 32-bit hashes, or in theory any other width).

Their structure is thus:

.. code-block:: C

    struct ExactIndexEntry {
        IntType key_hash;
        IntType key_ptr;
        IntType value;
    };

    struct ApproxIndexEntry {
        IntType key_hash;
        IntType value;
    };

    struct IndexHeader {
        uint64_t num_items;
        uint64_t index_ptr;
    };

    struct ExactIndex {
        IndexHeader header;
        char key_data[];
        ExactIndexEntry entries[];
    };

    struct ApproxIndex {
        IndexHeader header;
        ApproxIndexEntry entries[];
    };

In that structure, ``IntType`` can be any unsigned integer appropriate for the index "width" in use. Current
implementations use the same width for key hash, key pointer and value.

.. c:type:: IndexHeader

    .. attribute:: num_items

        Number of entries in the index

    .. attribute:: index_ptr

        Position of the index :attr:`entries` relative to the start of the ``IndexHeader``.

.. c:type:: ExactIndex

    .. attribute:: key_data

        Variable-length area where exact binary representations of the keys are stored in exact indexes.

    .. attribute:: entries

        The index entries, sorted by :attr:`key_hash`.

.. c:type:: ExactIndexEntry

    .. attribute:: key_ptr

        The position, relative to the start of the index header, of the key's value (should point inside :attr:`key_data`).

.. c:type:: ApproxIndexEntry

    .. attribute:: key_hash

        The hash value of the entry's key. The :attr:`entries` are sorted by this field.

    .. attribute:: value

        The value associated with the key. User-defined, but usually a position in some other data area,
        or index in a typed array, for all usages of :term:`Id Mapper` s within this library.

Key lookup
----------

Key lookup within the structure is performed with an optimized binary search to find the entries with the appropriate
hash.

For exact indexes, the actual keys can be compared to the lookup key by using the entries' :attr:`key_ptr`
field to look into :attr:`key_data` for the entries' actual key value to resolve hash collisions.
For approximate keys, all entries matching the :attr:`key_hash` are reported instead.

Simple mappers will return the first match, whereas multi-mappers would report all matches. They have no other
difference, so they are binary-compatible among themselves, if the data types and hash functions involved are the same.

.. _exponential search: https://en.wikipedia.org/wiki/Exponential_search
