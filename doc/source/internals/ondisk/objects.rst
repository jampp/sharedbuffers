Object structure
================

This section describes the format of all data built with `Schema.pack` and other similar `Schema` methods.

All object data is composed of a header, followed by fixed-size attributes, and then followed by variable-sized
data. Except for fixed-size attributes, all data is aligned to a configurable alignment size, by default 8 bytes.
Fixed-size attributes will be aligned to their native size instead, as long as the configured alignment size
is a multiple of that size (which it is, for the default setting).

.. code-block:: C

    struct ObjectHeader {
        BitmapType has_bitmap, none_bitmap;
    }

The ``BitmapType`` in the above definition depends on the number of attributes present on the object.
It will be an 8-bit integer for objects with up to 8 attributes, 16-bit for up to 16 attribute, 32-bit
for up to 32 attributes, and 64-bit for objects up to 64 attributes.

The format currently doesn't support a larger number of attributes.

Immediately after the object header, come the attribute values. The order of attributes is implicit,
defined by the object's `Schema`, and computed to guarantee alignment of each attribute to its
native word size. Basically, bigger attributes come first, followed by smaller attributes. For same-size
attributes, an arbitrary but stable order is chosen, currently alphabetically by attribute name.

:ref:`schema-serialization`, when performed, will make sure the attribute order is interpreted correctly even
across varying versions of the schema. It also persists serialization options, like alignment size and
other such configurable parameters.

Fixed size attributes include the primitive types: numbers (integer and float) and bool.
Non fixed-size attributes are encoded as a 64-bit pointer relative to the offset where the object header
is located.

Only attributes set and not None will be included in the attribute data, though. An attribute is set
if ``has_bitmap & (1 << attribute_position) != 0``. Similarly, an attribute is None if
``none_bitmap & (1 << attribute_position) != 0``, where ``attribute_position`` depends on the attribute
order computed by the rule mentioned above (or serialized in the persistent schema definition when
:ref:`schema-serialization` is performed).

After that data, padding is added to reach a properly aligned location, and then variable-sized attribute
data follows. Variable-sized attributes are appended after padding without any header, in arbitrary order.
They can be located through the relative pointers present in the attribute data section.

Objects *can* refer, through relative pointers, to any value stored in lower locations, so the
pointers may contain negative offsets in that case, and are thus signed.

.. _rtti-wrapping:

RTTI wrapping
-------------

Objects can be contained within objects, and sometimes the type (:class:`~sharedbuffers.mapped_struct.Schema`) of a
reference isn't known in advance. This happens for all container types when the type of the contents isn't uniform or
fixed in the :class:`~sharedbuffers.mapped_struct.Schema`, and for attribute references when the referent type is
annotated plainly as ``object``.

One can produce :term:`RTTI`-wrapped values on demand through `mapped_object.pack_into`, which takes objects of any
supported type, and produces the structure described in this section.

So, when necessary, values are wrapped in a header with :term:`RTTI`. The process is slightly different for
variable-sized types vs fixed-size types. In the case of variable-size types, the header is padded
to reach alignment, and the value is then appended as it normally would if the type were known right after
the padded header.

For fixed-size types, the header structure and the value itself are merged to achieve a more compact representation:

.. code-block:: C

    struct RTTIHeader {
        char typecode;
        union {
            int8_t i8;
            uint8_t ui8;
            int16_t i16;
            uint16_t ui16;
            int32_t i32;
            uint32_t ui32;
            int64_t i64;
            uint64_t ui64;
            float f;
            double d;
            bool b;
        }
    }

As can be gleaned from the above, the value itself won't be aligned except for the byte-sized types, and
for the wider types, the value could end up being bigger than the alignment size (in which case more padding will
be added to reach a multiple of it).

A C compiler would produce a struct large enough to accomodate all the entries in the union. In our DSL, we won't
concern ourselves with that, as readers won't care about struct size (no fixed-position field is present after the
union).

So, an :term:`RTTI`-wrapped uint8 would occupy 2 bytes of data plus 6 bytes of padding (for the default alignment of 8),
a uint32 would instead have 5 bytes of data plus 3 bytes of padding, and a double would contain 9 bytes of data
followed by 7 bytes of padding.

Padding is only added to make sure objects following this one remain aligned, but serves no other purpose.

.. important::

    Pointers to wrapped values aren't valid pointers to unwrapped values, but it may be possible to use a pointer
    to the wrapped value inside (using an offset pointer skipping the :term:`RTTI` header), as a bare unwrapped
    pointer, if the value isn't one of the fixed-size built-ins.

Typecodes for built-in types include:

===========  ================================================
typecode     type
===========  ================================================
B            uint8
b            int8
H            uint16
h            int16
I            uint32
i            int32 / int / long
Q            uint64
q            int64
f            float32
d            float64 / float
T            bool
Z            :ref:`frozenset <frozenset-internals>`
t            :ref:`tuple <sequence-internals>`
e            :ref:`list <sequence-internals>`
s            :ref:`bytes <bytes-internals>`
u            :ref:`unicode <unicode-internals>`
r            :ref:`buffer <buffer-internals>`
m            :ref:`dict <dict-internals>`
v            :ref:`datetime <datetime-internals>`
V            :ref:`date <datetime-internals>`
F            :ref:`Decimal <decimal-internals>`
E            :ref:`proxied_list <sequence-internals>`
W            :ref:`proxied_tuple <sequence-internals>`
n            :ref:`ndarray <ndarray-internals>`
M            :ref:`proxied_dict <dict-internals>`
z            :ref:`proxied_frozenset <frozenset-internals>`
===========  ================================================

Custom types can be registered to custom typecodes, through `mapped_object.register_schema`. Typecodes with codes
above ``0x80`` are reserved for that purpose.

Examples
--------

Given the following Python class:

.. code-block:: Python

    class SomeStruct(object):
        def __init__(self, **kw):
            self.__dict__.update(kw)

    SomeStruct.__slot_types__ = {
        'smallnum': int8,
        'bignum': int64,
        'fraction': float32,
        'bigger_fraction': float64,
        'otherstruct': SomeStruct,
        'yetanother': object,
    }

    SomeStruct.__schema__ = Schema.from_typed_slots(SomeStruct)
    mapped_object.register_schema(SomeStruct, SomeStruct.__schema__, '\x80')
    SomeStruct.__schema__.reinitialize()

    some_data = SomeStruct(
        smallnum=3,
        bignum=12341234,
        fraction=1.5,
        bigger_fraction=3.14,
        otherstruct=SomeStruct(
            smallnum=2,
            bignum=1234,
            yetanother=None,
        ),
        yetanother=SomeStruct(
            smallnum=2,
            bignum=1234,
            bigger_fraction=7.28,
            otherstruct=None,
        ),
    )

If we were to pack ``some_data`` with `Schema.pack`, we'd get:

.. code-block:: pycon

    >>> SomeStruct.__schema__.pack(some_data)
    bytearray(
        b"?\x00\x1f\x85\xebQ\xb8\x1e\t@\xf2O\xbc\x00\x00\x00\x00\x00(\x00"
        b"\x00\x00\x00\x00\x00\x008\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0"
        b"?\x03\x00*\x08\xd2\x04\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00"
        b"\x00\x80\x00\x00\x00\x00\x00\x00\x00\'\x04\x1f\x85\xebQ\xb8\x1e"
        b"\x1d@\xd2\x04\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00")

:ref:`schema-serialization` gives the following state for this schema (excluding ``slot_types`` which we know already):

.. code-block:: Python

    {
        'slot_types': {...},
        'slot_keys': (
            'bigger_fraction', 'bignum', 'otherstruct', 'yetanother', 'fraction', 'smallnum'
        ),
        'bases': None,
        'alignment': 8
    }

This means attributes will be stored in the order given by ``slot_keys`` which, as noted earlier,
sorts attributes from biggest to smallest.

Looking more closely, and parsing that structure according to the schema defined above, we get:

.. code-block:: C

    struct OtherStruct {
        unsigned char has_bitmap, none_bitmap;

        long long bignum;
        char smallnum;

        char padding[5];
    }

    struct YetAnother {
        unsigned char has_bitmap, none_bitmap;

        double bigger_fraction;
        long long bignum;
        char smallnum;

        char padding[5];
    }

    struct YetAnotherWrapped {
        char typecode;
        char padding[7];
        YetAnother value;
    }

    struct SomeStruct {
        unsigned char has_bitmap, none_bitmap;

        double bigger_fraction;
        long long bignum;
        long long otherstruct, yetanother;
        float fraction;
        char smallnum;

        char padding;

        OtherStruct otherstruct_value;
        YetAnotherWrapped yetanother_value;
    }

Which gives us

.. code-block:: C

    SomeStruct some_struct = {
        .has_bitmap = 0x3f,
        .none_bitmap = 0,

        .bigger_fraction = 3.14,
        .bignum = 12341234,
        .otherstruct = 40,
        .yetanother = 56,
        .fraction = 1.5f,
        .smallnum = 3,

        .padding = 0,

        .otherstruct_value = {
            .has_bitmap = 0x2a,
            .none_bitmap = 8,

            .bignum = 1234,
            .smallnum = 2,

            .padding = {0,0,0,0,0}
        },
        .yetanother_value = {
            .typecode = 0x80,
            .padding = {0,0,0,0,0,0,0},
            .value = {
                .has_bitmap = 0x27,
                .none_bitmap = 4,

                .bigger_fraction = 7.28,
                .bignum = 1234,
                .smallnum = 2,

                .padding = {0,0,0,0,0}
            }
        }
    };

In essence, for each object of the given `Schema`, the ``has_bitmap`` entirely defines the shape of the C-level
struct that holds the attribute data, whereas the ``none_bitmap`` specifies, for missing attributes, whether they're
actually missing or do they contain implicit ``None`` values.
