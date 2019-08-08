.. _sequence-internals:

Sequence types
==============

Both lists and tuples share the same binary representation and are thus binary-compatible.

.. code-block:: C

    union VarLength {
        // For typecodes l, L, d, t, T
        long longlength:56;

        // For other typecodes, length < 0xFFFFFF
        int shortlength:24;

        struct {
            int padding:24; // 0xFFFFFF
            int64_t length;
        } verylong;
    };

    struct List {
        char typecode;
        VarLength length;
        ListData data;
    };

    union ListData {
        // Integral data
        uint8_t ui8[];  // typecode B
        int8_t i8[];  // typecode b
        uint16_t ui16[];  // typecode H
        int16_t i16[];  // typecode h
        uint32_t ui32[];  // typecode I
        int32_t i32[];  // typecode i
        uint64_t ui64[];  // typecode Q
        int64_t i64[];  // typecode q

        // Float data
        double f64[];  // typecode d

        // Relative pointers to RTTI-wrapped data
        int32_t p32[];  // typecode T
        int64_t p64[];  // typecode t
    };

As always with unions, they take as much space as needed by actually present fields, rather than using
up as many bytes as the largest field. Which means a short header is exactly 4 bytes, a medium header
8 bytes, and the longest possible header is 16 bytes.

Integer data is stored in-line in a compact format. Narrow integer types (32 bit and less) may use short
headers if sequence length allows it. Wider integer and float types always use medium headers. In this
way alignment is always guaranteed to at least native word boundaries.

Integer data representation is limited for sequences of uniform type. If there's a mix of types, or if
data isn't numeric at all, pointers relative to the list's header position will be stored instead.
To get a more compact representation in the most common cases, when those pointers are narrow enough
(ie: when the actual object data is located close enough for 32-bit signed pointers), a 32-bit pointer
table is used instead.

Data pointed to by both narrow and wide pointer tables must be wrapped in :term:`RTTI` since there's
no implicit type information in sequence types. ``None``, as a rather common value, is represented by
the special pointer ``1``, which can't ever be a valid relative pointer (it would point inside the
sequence header).

.. important::

    While the ``1`` pointer is reserved for ``None`` since it is otherwise invalid, the zero is also
    invalid, but only due to a subtle distinction. ``0`` can actually refer to the sequence itself,
    which is in fact a valid sequence with a recursive reference. But to be a valid pointer inside
    a sequence, it must point to the :term:`RTTI` header, which can never be at offset ``0``, but
    rather ``-8``.

    It should **not** be assumed that ``0`` will always be invalid, as it is desirable to be able
    to define statically-typed sequence types, and may in fact be introduced in future versions.

Additionally, there are a couple typecodes used by sets and not lists. See :ref:`frozenset-internals`.

Summarizing:

===========  ================================================
typecode     type
===========  ================================================
B            uint8
b            int8
H            uint16
h            int16
I            uint32
i            int32
Q            uint64
q            int64
d            float64
t            int32 relative pointers
T            int64 relative pointers
m            56-bit bitmap
M            120-bit bitmap
===========  ================================================

.. _frozenset-internals:

Frozensets
==========

Frozensets are represented on-disk as sorted tuples without duplicates, except for very small sets
of small non-negative numbers, case in which extra typecodes ``m`` and ``M`` are used, whose representation are
instead small bitmaps.

So, for any other typecode, frozensets are represented as sorted tuples, and can be read as such.

Numeric sets are sorted numerically. Object sets, however, are sorted by their `_stable_hash` value.

When the typecode is ``m``, a 56-bit bitmap is used. ``ListData`` for typecode ``m`` thus contains 7 8-bit
unsigned integers (``ui8`` is of length 7). For typecode ``M``, the bitmap is 120-bit wide instead (``ui8`` is
of length 15).

If the frozenset is ``fset``, and ``0 <= x < 120 or 56``, the contents of ``ui8`` are defined as:

    ``ui8[x / 8] & (1 << (x % 8)) != 0`` iff ``x in fset``

Examples
--------

Small numeric sequences
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_tuple.pack_into((1,3,7,20), buf, 0)
    >>> buf[:end]
    bytearray(b'B\x04\x00\x00\x01\x03\x07\x14')

    >>> end = mapped_tuple.pack_into((1,3,7,20,8777), buf, 0)
    >>> buf[:end]
    bytearray(b'H\x05\x00\x00\x01\x00\x03\x00\x07\x00\x14\x00I"\x00\x00')

    >>> end = mapped_tuple.pack_into((1,3,7,20,87770000), buf, 0)
    >>> buf[:end]
    bytearray(
        b'i\x05\x00\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x07\x00\x00\x00\x14\x00\x00\x00\x90C;\x05')

In these examples, we see short headers with length 4, 5, and 5 respectively, and their contents as packed
``uint8``, ``uint16`` and ``int32`` arrays. Nothing too fancy.

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_frozenset.pack_into([1, 3, 1<<40], buf, 0)
    >>> buf[:end]
    bytearray(
        b'q\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00'
        b'\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00')

This one contains ``int64`` arrays and uses a medium header with length 3. A medium header is forced since the
data type requires 8-byte alignment anyway.

Object sequences
~~~~~~~~~~~~~~~~

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_tuple.pack_into((1,3,7,20,None), buf, 0)
    >>> buf[:end]
    bytearray(
        b'T\x05\x00\x00\x00\x00\x00\x00 \x00\x00\x000\x00\x00\x00'
        b'@\x00\x00\x00P\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
        b'q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')

This example is a tad more complex. We can see here a medium header again, with a 32-bit relative pointer table
pointing to 4 values, and an implicit ``None``. Parsing this buffer into a C-like struct would give us:

.. code-block:: C

    struct {
        List list;
        char padding[4];
        RTTIHeader val_1, val_3, val_7, val_20;
    } list_value = {
        .list = {
            .typecode = 'T',
            .length = {
                .longlength = 5
            },
            .data = {
                .p32 = {32, 48, 64, 80, 1}
            }
        },

        // Padding to reach 8-byte alignment
        .padding = {0,0,0,0},

        // Offset 32 here
        .val_1 = {.typecode = 'q', .i64 = 1},
        .val_3 = {.typecode = 'q', .i64 = 3},
        .val_7 = {.typecode = 'q', .i64 = 7},
        .val_20 = {.typecode = 'q', .i64 = 20},
    }

A slightly more complex example can be obtained by building a recursive reference into the sequence:

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> l = [1, 3, 7, 20]
    >>> l.append(l)
    >>> end = mapped_list.pack_into(l, buf, 0)
    >>> buf[:end]
    bytearray(
        b'T\x05\x00\x00\x00\x00\x00\x00 \x00\x00\x000\x00\x00\x00'
        b'@\x00\x00\x00P\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00'
        b'q\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'q\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'e\x00\x00\x00\x00\x00\x00\x00'
        b'T\x05\x00\x00\x00\x00\x00\x00'
        b'\xb8\xff\xff\xff\xc8\xff\xff\xff\xd8\xff\xff\xff\xe8\xff\xff\xff'
        b'\xf8\xff\xff\xff\x00\x00\x00\x00')

Parsing this into C-like structures shines a light into a few interesting points:

.. code-block:: C

    struct {
        List list;
        char padding[4];
        RTTIHeader val_1, val_3, val_7, val_20;
        struct {
            RTTIHeader rtti;
            char padding[7];
            List value;
            char padding[4];
        } val_reclist;
    } recursive_list_value = {
        .list = {
            .typecode = 'T',
            .length = {
                .longlength = 5
            },
            .data = {
                .p32 = {32, 48, 64, 80, 96}
            }
        },

        // Padding to reach 8-byte alignment
        .padding = {0,0,0,0},

        // Offset 32 here
        .val_1 = {.typecode = 'q', .i64 = 1},
        .val_3 = {.typecode = 'q', .i64 = 3},
        .val_7 = {.typecode = 'q', .i64 = 7},
        .val_20 = {.typecode = 'q', .i64 = 20},

        // Offset 96 here
        .val_reclist = {
            .rtti = {.typecode = 'e'},
            .padding = {0,0,0,0,0,0,0},
            .value = {
                .typecode = 'T',
                .length = {
                    .longlength = 5
                },
                .data = {
                    .p32 = {-72, -56, -40, -24, -8}
                }
            },
            .padding = {0,0,0,0}
        }
    }

One can't help but notice 2 lists in this structure, instead of just 1.

That's because we first packed the list as a non-:term:`RTTI`-wrapped value at offset 0.
But, in this way, when we get to the recursive reference, there's no way to encode that pointer,
since there's no :term:`RTTI` header anywhere in the buffer for the value at offset 0 (there can't be,
it would have to be at offset ``-8``, before the buffer's starting point).

So a second copy of the list is packed, this time wrapped in :term:`RTTI`, and reusing references to all the contents.
All pointers here are negative pointers pointing to already-packed objects, including the recursive reference itself,
which can now reference to this second copy by its :term:`RTTI` header at offset 96 (``-8`` relative to itself).

If this list was embedded into a larger structure that already required :term:`RTTI` for the first reference, this
second copy would not occur, and the ``-8`` pointer would instead be used from the start.

This clearly illustrates how pointers to :term:`RTTI`-tagged objects can't be used in place of "bare" pointers,
and viceversa. Doing so would result in a broken, unparseable structure.

It also shows how :term:`RTTI`-tagged numbers are considerably more verbose than bare numbers, and thus why
specialized representations for numeric sequences are worth the effort.

Frozensets
~~~~~~~~~~

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_frozenset.pack_into(set(), buf, 0)
    >>> buf[:end]
    bytearray(b'm\x00\x00\x00\x00\x00\x00\x00')

    >>> end = mapped_frozenset.pack_into(set([1,7,20]), buf, 0)
    >>> buf[:end]
    bytearray(b'm\x82\x00\x10\x00\x00\x00\x00')

    >>> end = mapped_frozenset.pack_into(set([1,7,20,66]), buf, 0)
    >>> buf[:end]
    bytearray(b'M\x82\x00\x10\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00')

These 3 examples showcase bitmap representations, both short and long. As can already
be seen in the bytearray's repr output:

* The first example contains no bits set, which represents an empty set
* The second example contains 3 bits set, 2 in the first byte (bits 1 and 7), and 1 on the third byte (bit 4, which
  in that position corresponds to bit 20).
* The third example shows a bigger bitmap with bits 1, 7, 20 and 66, spread in 3 bytes.

Notice that all bit positions start in 0, so bit 1 is the second bit.

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_frozenset.pack_into(set([1,1875,7,20,66]), buf, 0)
    >>> buf[:end]
    bytearray(b'H\x05\x00\x00\x01\x00\x07\x00\x14\x00B\x00S\x07\x00\x00')

This last example is a straightforward case of a set represented by a sorted list.

Finally:

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)
    >>> end = mapped_frozenset.pack_into(set(['foobar', None, 'barbaz']), buf, 0)
    >>> buf[:end]
    bytearray(
        b'T\x03\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00'
        b'\x18\x00\x00\x00(\x00\x00\x00\x00\x00\x00\x00'
        b's\x00\x00\x00\x00\x00\x00\x00\x06\x00barbaz'
        b's\x00\x00\x00\x00\x00\x00\x00\x06\x00foobar')

This is a more complex example, where the set is represented also by a sorted list. But the underlying
list here is ``[None, 'barbaz', 'foobar']``, whose order is defined by the `_stable_hash` of the items
instead of the items themselves.

Parsing this as a C-like struct, we get:

.. code-block:: C

    struct {
        List list;
        char padding[4];
        struct {
            RTTIHeader rtti;
            char padding[7];
            BytesData value;
        } val_barbaz;
        struct {
            RTTIHeader rtti;
            char padding[7];
            BytesData value;
        } val_foobar;
        RTTIHeader val_barbaz, val_foobar;
    } set_value = {
        .list = {
            .typecode = 'T',
            .length = {
                .longlength = 3
            },
            .data = {
                .p32 = {1, 24, 40}
            }
        },

        // Padding to reach 8-byte alignment
        .padding = {0,0,0,0},

        // Offset 24 here
        .val_barbaz = {
            .rtti = {.typecode = 's'},
            .padding = {0,0,0,0,0,0,0},
            .value = {
                .length = {.shortlen = 6},
                .data = "barbaz",  // no null terminator
            },
        }

        // Offset 40 here
        .val_barbaz = {
            .rtti = {.typecode = 's'},
            .padding = {0,0,0,0,0,0,0},
            .value = {
                .length = {.shortlen = 6},
                .data = "foobar",  // no null terminator
            },
        }
    }
