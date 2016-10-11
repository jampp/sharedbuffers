sharedbuffers
=============

This library implements shared-memory typed buffers that can be read and manipulated (and we'll eventually 
support writes too) efficiently without serialization or deserialization.

The main supported implementation of obtaining shared memory is by memory-mapping files, but the library also supports
mapping buffers (anonymous mmap objects) as well, albeit they're harder to share among processes.

Currently, most primitive types and collections are supported, except `dicts`.

Supported primivite types:

    * int (up to 64 bit precision)
    * str (bytes)
    * unicode
    * frozenset
    * tuple / list

Primitive types are cloned into their actual builtin objects when accessed. Although fast, it does imply that contianers
will take up a lot of process-local memory when accessed. Support for collection proxies that take the place of
builtin containers is planned for a future release.

Objects can be registered with schema serializers and thus composite types can be mapped as well. For this to function
properly, objects need a class attribute specifying the attributes it holds and the type of the attributes. When an
attribute doesn't have a clearly defined type, it can be wrapped in a RTTI-containing container by specifying it as
type `object`.

For example:

.. code:: python

    class SomeStruct(object):
        __slot_types__ = {
            'a' : int,
            'b' : float,
            's' : str,
            'u' : unicode,
            'fset' : frozenset,
            'l' : list,
            'o' : object,
        }
        __slots__ = __slot_types__.keys()

Adding `__slot_types__`, however, isn't enough to make the object mappable. A schema definition needs to be created,
which can be used to map files or buffers and obtain proxies to the information within:

.. code:: python

    class SomeStruct(object):
        __slot_types__ = {
            'a' : int,
            'b' : float,
            's' : str,
            'u' : unicode,
            'fset' : frozenset,
            'l' : list,
            'o' : object,
        }
        __slots__ = __slot_types__.keys()
        __schema__ = mapped_struct.Schema.from_typed_slots(__slot_types__)

Using the schema is thus straightforward:

.. code:: python

    s = SomeStruct()
    s.a = 3
    s.s = 'blah'
    s.fset = frozenset([1,3])
    s.o = 3
    s.__schema__.pack(s) # returns a bytearray

    buf = bytearray(1000)
    s.__schema__.pack_into(s, buf, 10) # writes in offset 10 of buf, returns the size of the written object
    p = s.__schema__.unpack_from(s, buf, 10) # returns a proxy for the object just packed into buf, does not deserialize
    print p.a
    print p.s
    print p.fset

Typed objects can be nested, but for that a typecode must be assigned to each type in order for RTTI to properly
identify the custom types:

.. code:: python

    SomeStruct.__mapped_type__ = mapped_struct.mapped_object.register_schema(
        SomeStruct, SomeStruct.__schema__, 'S')

From then on, `SomeStruct` can be used as any other type when declaring field types.

High-level typed container classes can be created by inheriting the proper base class. Currently, there are only
arrays and mappings of two kinds: string-to-object, and uint-to-object

.. code:: python

    class StructArray(mapped_struct.MappedArrayProxyBase):
        schema = SomeStruct.__schema__
    class StructNameMapping(mapped_struct.MappedMappingProxyBase):
        IdMapper = mapped_struct.StringIdMapper
        ValueArray = StructArray
    class StructIdMapping(mapped_struct.MappedMappingProxyBase):
        IdMapper = mapped_struct.NumericIdMapper
        ValueArray = StructArray

The API for these high-level container objects is aimed at collections that don't really fit in RAM in their
pure-python form, so they must be built using an iterator over the items (ideally a generator that doesn't
put the whole collection in memory at once), and then mapped from the resulting file or buffer. An example:

.. code:: python

    with tempfile.NamedTemporaryFile() as destfile:
        arr = StructArray.build([SomeStruct(), SomeStruct()], destfile=destfile)
        print arr[0]

    with tempfile.NamedTemporaryFile() as destfile:
        arr = StructNameMapping.build(dict(a=SomeStruct(), b=SomeStruct()).iteritems(), destfile=destfile)
        print arr['a']

    with tempfile.NamedTemporaryFile() as destfile:
        arr = StructIdMapping.build({1:SomeStruct(), 3:SomeStruct()}.iteritems(), destfile=destfile)
        print arr[3]

When using nested hierarchies, it's possible to unify references to the same object by specifying an idmap dict.
However, since the idmap will map objects by their `id()`, objects must be kept alive by holding references to
them while they're still referenced in the idmap, so its usage is non-trivial. An example technique:

.. code:: python

    def all_structs(idmap):
        iter_all = iter(some_generator)
        while True:
            idmap.clear()
    
            sstructs = list(itertools.islice(iter_all, 10000))
            if not sstructs:
                break
    
            for ss in sstructs :
                # mapping from "s" attribute to struct
                yield (ss.s, ss)
            del sstructs
    
    idmap = {}
    name_mapping = StructNameMapping.build(all_structs(idmap), 
        destfile = destfile, idmap = idmap)

The above code syncs the lifetime of objects and their idmap entries to avoid mapping issues. If the invariant
isn't maintained (objects referenced in the idmap are alive and holding a unique `id()` value), the result will be
silent corruption of the resulting mapping due to object identity mixups.

There are variants of the mapping proxy classes and their associated id mapper classes that implement multi-maps.
That is, mappings that, when fed with multiple values for a key, will return a list of values for that key rather
than a single key. Their in-memory representation is identical, but their querying API returns all matching values
rather than the first one, so multi-maps and simple mappings are binary compatible.

Multi-maps with string keys can also be approximate, meaning the original keys will be discarded and the mapping will
only work with hashes, making the map much faster and more compact, at the expense of some inaccuracy where the
returned values could have extra values corresponding to other keys whose hash collide with the one being requested.
