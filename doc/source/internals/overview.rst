Conceptual overview
===================

.. _schema-portability:

Portability
-----------

Shared buffers are stored in-memory (and on-disk) as almost-portable binary structures, their portability only
limited by their dependence on a previously agreed `Schema`, and the fact that numbers are stored in native
endianness and format, making those buffers platform-dependent.

Between compatible platforms and code, they are fully portable. The binary representation is well defined
and language-agnostic, which makes it parseable from multiple languages (as long as a suitable implementation
is available).

Compatibility-breaking changes are infrequent, but not guaranteed not to occur at least until version 1.0.0 is
released. After version 1.0.0, any changes that break binary compatibility across otherwise
:ref:`compatible schemas <schema-compatibility>` will include a major version and "magic number" bump (meaning the
library should raise an error rather than misinterpret incompatible data).

Before version 1.0.0, an effort is made to detect garbage or incompatible data formats, but not all such instances
are guaranteed to fail gracefully.

.. _schema-compatibility:

Schema compatibility
--------------------

Schemas are considered fully binary compatible if they have the exact same slots and slot types, stored in the
same order, with the same alignment parameters and typecode registrations for relevant types.

Since version 0.4.0, slot order is deterministic, so schemas that share the same slot types and typecode
registrations should always be compatible with one another.

Schemas with differing slot types can be binary compatible if the in-memory representation of the slots
is compatible instead. Some types differ only in the API used to expose the information rather than their
in-memory representation. For example, tuples and lists are binary-compatible, so switching from one type
to the other would not break schema compatibility. See each type's internals documentation for details on
which types are binary-compatible in that fashion.

.. important::

    Statically typed attributes don't require :term:`RTTI` wrapping, whereas dynamically typed attributes
    do. As such, even if types are binary-compatible or even equal, switching from static typing to
    dynamic typing and viceversa is **not** a binary-compatible change.

:ref:`schema-serialization` is a way to guarantee buffers are interpreted with the right schema, under
some circumstances, by serializing the `Schema` itself into the shared buffer, and using it to reconstruct
the original schema used when building the buffer, regardless of which version of the schema is used
when reading the buffer.

Code-level compatibility then is up to the application and language in use. In Python, extra attributes are
harmless (just ignored) and missing attributes just raise `AttributeError` when accessed, as a normal python
object would if it were missing an attribute. If the application can handle that, then the buffer will be compatible.

More details can be found on the :ref:`schema-serialization` section.