String types
============

.. _bytes-internals:

Bytes
-----

Byte strings are stored as a variable-length size + data.

.. code-block:: C

    #define COMPRESSED 0x8000

    typedef short ShortLength;

    struct LongLength {
        ShortLength shortlen;  // with length set to 0x7fff for long-length values
        unsigned long long length;
    };

    union VarLength {
        ShortLength shortlen;
        LongLength longlen;
    }

    struct BytesData {
        VarLength length;
        char[] data;
    };

As always with unions in our C-like DSL, the union structure ``VarLength``'s size is the minimum needed for the case,
instead of the maximum as a normal C compiler would. So a ShortLength would only use 2 bytes, whereas a LongLen would
use 10 bytes.

When reading, a reader should try to read the header as a ShortLength. If the length value is exactly ``0x7fff``,
then a LongLength should be parsed instead.

These values can be constructed through `mapped_bytes.pack_into`.

The ``ShortLength`` value embeds a ``compressed`` bit on its MSB, so it's largest length is ``0x7fff``.
When the ``compressed`` bit is set (ie, ``length >= 0x8000``), byte ``data`` is compressed with LZ4.

.. _unicode-internals:

Unicode
-------

Unicode strings are stored as UTF-8 :ref:`byte strings <bytes-internals>`. They are thus binary-compatible,
as long as the byte string contents are valid UTF-8.

.. _buffer-internals:

Buffers
-------

Buffers are straightforwardly stored as a length + data. Nothing fancy:

.. code-block:: C

    struct BufferData {
        unsigned long long length;
        char[] data;
    };

Buffer data is never compressed (otherwise they wouldn't be efficiently accessible).

These can be constructed with `proxied_buffer.pack_into`.

Examples
--------

.. code-block:: pycon

    >>> buf = bytearray(1 << 20)

    >>> end = mapped_bytes.pack_into("foobar", buf, 0)
    >>> buf[:end]
    bytearray(b'\x06\x00foobar')

    >>> end = mapped_unicode.pack_into(u"fóöbær", buf, 0)
    >>> buf[:end]
    bytearray(b'\t\x00f\xc3\xb3\xc3\xb6b\xc3\xa6r')

    >>> end = proxied_buffer.pack_into(buffer("barfoo"), buf, 0)
    >>> buf[:end]
    bytearray(b'\x06\x00\x00\x00\x00\x00\x00\x00barfoo')

    >>> end = mapped_bytes.pack_into("foobar" * 1024, buf, 0)
    >>> end
    45
    >>> buf[:end]
    bytearray(
        b'+\x80\x00\x18\x00\x00ofoobar\x06\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff'
        b'\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xf9Poobar')

    >>> end = mapped_bytes.pack_into(os.urandom(128 << 10), buf, 0)
    >>> end
    131082
    >>> buf[:20]
    bytearray(b'\xff\x7f\x00\x00\x02\x00\x00\x00\x00\x00\xcc\xd1\x95#\x979\xf5\xf4l\x17')

Inspecting header structure, we can see:

* Case 1: a ``ShortLength`` with value ``6``, uncompressed, byte data literally ``foobar`` in ASCII.
* Case 2: a ``ShortLength`` with value ``9``, uncompressed, byte data UTF-8 encoded.
* Case 3: a 64-bit length with value ``6``, followed by literal byte data (``barfoo`` in ASCII in this case).
* Case 4: a ``ShortLength`` with value ``43``, followed by LZ4-compressed data.
* Case 5: a ``LongLength`` with the ``compress`` bit unset (``shortlen == 0x7fff``), and ``length=0x20000``,
  followed by literal uncompressed byte data.

