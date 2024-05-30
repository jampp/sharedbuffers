Change Log
..........

All notable changes to this project will be documented here.

The format is largely inspired by keepachangelog_.

v1.2.2 - Unreleased
===================

Bugfixes
~~~~~~~~

- Fix a bug when packing big objects (bigger than the default 1MB
  pack buffer) into buffer pools when it requires adding a new section.
  It would fail to use a properly sized pack buffer, raising an error
  instead of succeeding as expected.

v1.2.1 - 2023-10-10
===================

Bugfixes
~~~~~~~~

- Fix a bug in which proxied frozensets using the long bitmap representation
  would suffer from integer overflow and essentially wipe half the entries.

v1.2.0 - 2023-08-17
===================

Added
~~~~~

- Schema can now take arbitrary ``user_data`` that will be serialized
  when the schema is serialized. This can be useful to be able to cleanly
  embed metadata about the schema in the schema itself.
- Mappings' iteritems now can take ``g_kw`` and ``idm_kw`` to be able
  to customize idmap and getter kwargs. This can be useful to accelerate
  or otherwise customize iteration.
- Mappings' build method now accepts a ``value_cache_size`` argument to
  customize the cache used to de-duplicate values. As a special case, if
  this value is 0, de-duplication is deactivated. This is useful to simplify
  the way simple mappings are constructed when de-duplication is not needed.
  If de-duplication is enabled and the objects are simple enough, the
  careful lifetime management documented is no longer necessary, a significant
  quality of life improvement.
- Id mappers now expose an ``index_data`` attribute to access the index data
  itself, which can empower advance manipulation of the mappers without having
  to fully rebuild them.
- Id mappers can now be copied. This will create a writable copy of the id mapper
  and can be useful to create derived mappers, but be warned that it does load
  the whole mapper's contents into process memory.

Bugfixes
~~~~~~~~

- Fixed a bug in timezone calculations with python 3

v1.1.1 - 2023-06-21
===================

Bugfixes
--------

- Fixes some compatibility issues with newer python 3 and numpy versions
- Some general python 3 compatibility fixes (keys/items/values should
  return iterators when mimicking dicts)
- Work around an issue with python 3's memoryview GC behavior which could
  segfault in some contrived cases, by actively breaking cycles in proxy
  buffer object finalizers.
- Implement complete support for native float32 lists. Support was there
  already but only partial, and not properly exercised due to a bug.
- Invoke base class finalizers from proxy buffer object finalizers.
- Catch and discard mmap.close() exceptions when finalizing buffer pools.
  Dereferencing them with active references to the underlying data is
  all too common, but actively closing the mmap when possible is still
  desirable.

v1.1.0 - 2023-05-24
===================

Added
~~~~~

- Timezone-aware ``datetime`` type variants:

  - ``mapped_datetime_local`` will treat naive datetimes as local time and
    convert accordingly to UTC for storage, akin to automatically storing
    ``dt.replace(tzinfo=tzlocal())``.
  - ``mapped_datetime_utc`` will treat naive datetimes as UTC, akin to automatically
    storing ``dt.replace(tzinfo=tzutc())``.
  - ``mapped_datetime_tz`` datetime with timezone that treats naive datetimes as UTC,
    and stores the original timezone so it can be preserved upon reading
  - ``mapped_datetime_tz_local`` datetime with timezone that treats naive datetimes as
    local time.

Changed
~~~~~~~

- The behavior of ``datetime`` with naive datetime objects should now make more sense,
  but if an application made use of mixed or non-UTC datetime objects, it may differ from
  version 1.0.0 in an application-breaking way. It is however recommended that a change to
  an explicitly specified assumption or timezone be made, either by using
  one of ``mapped_datetime_local|utc|tz``.

Bugfixes
--------

- Pack timestamps used for ``date`` and ``datetime`` objects in UTC
- Fix GenericFileMapper's base implementation for zip files in python 3.10
- Fix ObjectIdMapper's base implementation for zip files in python 3.10

Technical Tasks
---------------

- Add requirements to run tests with pytest

v1.0.0 - 2022-08-23
===================

Changed
-------

- BTS-90_: Migrate sharedbuffers to Python 3
- BTS-619_: Remove pure python stuff

Bugfixes
--------

- Fix buffer acquisition error handling to prevent the possibility of
  double-release or other buffer reference mismanagement if an error
  happens during the GetBuffer call.

Technical Tasks
---------------

- BTS-618_: Make mmap ``flags = MAP_SHARED`` explicit to avoid confusions

v0.9.1 - 2021-03-25
===================

Changed
-------

- Optimize assuming the package will run in the platform doing the building
  without jeopardizing compatibility with other platforms by default.
  More precisely this involves passing ``-mtune=native`` to GCC by default.

v0.9.0 - 2020-02-25
===================

Major
-----

- Broken compatibility with buffers generated with earlier versions because
  of an alignment bugfix that changes the way objects are encoded. Sadly there's
  only a minimal backward compatibility mechanism in place so it's a binary
  compatibility break. Specifically, pickled schemas will contain a version
  number that will enable reading old buffers from pickled schemas, but for this
  to work all object references must be explicitly typed to allow all schemas
  to be pickled properly. If at all possible, it's safer to rebuild all buffers
  with 0.9.0 after the update.

Bugfixes
--------

- Fix edge case in binary search where if the skip search lands in the middle
  of a run of matching keys, the low search bound may be set such that it
  fails to return the actual first match
- Fix object attribute alignment by enlarging field bitmaps to match alignment.
  Fixes support for ARM architectures that really depend on alignment, but breaks
  compatibility with buffers generated by older versions.

v0.8.2 - 2019-09-12
===================

Improvements
------------

- Implement accurate hints for NumericIdMapper and derivatives, getting
  the improved access patterns of the StringIdMapper working on all mappers.

v0.8.1 - 2019-09-11
===================

Bugfixes
--------

- Fix MappedMappingProxyBase deduplication. It was scrambling values
  if duplicates were not contiguous.

v0.8.0 - 2019-08-08
===================

Added
-----

- Added extensive documentation about the binary format of
  shared buffers and other internals.

Changed
-------

- ApproxStringIdMultiMapper now also allows already hashed keys (integer keys)

Bugfixes
--------

- Fix requirements to exclude numpy 1.17.0 and above,
  as they are python 3.5+ only

v0.7.2 - 2019-06-27
===================

Bugfixes
--------

- Fix mapped_object to actually pick the smallest integer typecode possible
  instead of always picking 'q'. Also fix it for 'Q', which wasn't even
  working when the number didn't fit in 'q'.
- Fix mapped_tuple pack_into. If long headers were used, it would
  unintentionally expand the given buffer.

v0.7.1 - 2019-06-13
===================

Bugfixes
--------

- Fix read-write mappings to properly map as a shared memory segment.

v0.7.0 - 2019-06-06
===================

Added
-----

- Allow writing to numeric proxy properties when the mapping
  was specified as read-write.

- Allow atomic transactions for numeric proxy properties.

v0.6.4 - 2019-05-30
===================

Bugfixes
--------

- Disallow registering schemas with typecodes that conflict
  with builtin types' typecodes. This would cause all sorts
  of subtle breakage.

v0.6.3 - 2019-05-30
===================

This release was removed from pypi due to a build error.
It's otherwise identical to 0.6.4

v0.6.2 - 2019-02-21
===================

Bugfixes
--------

- Map only requested size, when specified, in id mappers.

v0.6.1 - 2019-01-23
===================

Bugfixes
--------

- Fix error trying to access None items within proxied_lists.
  If a proxied_list contained None items, they'd raise an
  inconsistent data exception on access even though there was
  no data inconsistency.

v0.6.0 - 2018-12-27
===================

Added
-----

- Allow reusing object pool sections, which is useful
  when repeatedly creating them (say, in tests)

v0.5.1 - 2018-12-17
===================

Bugfixes
--------

- Stable hashing now works with negative and 'special' floats.
- Fix access to empty id mappers that would either throw an
  error or, worse, if cythonized, segfault.

v0.5.0 - 2018-12-11
===================

Notable changes
---------------

- Added support for proxied variants of a number of collection
  classes that have constant property access times (item
  lookup times vary from class to class, but usually `O(log N)`)
- Proxies can now inherit from custom classes, so you can have
  proxies with behavior (methods and properties) that match
  the proxied object's
- Various optimizations across the board. A huge optimization
  effort improved both access and pack times for most cases.
- Added some low-level utilities to facilitate incremental
  construction of large object graphs.
- Now requires Cython 0.29 (when using Cython)
- Added support for quite a few extra primitive types.
- More compact representations of lists and sets in most cases.

Added
-----

- Add support for proxied buffers
- Add support for proxied numpy arrays
- Add support for proxied list/tuple/frozenset
- Add support for mapped dict
- Allow specifying custom bases for proxy objects, so proxies
  can inherit their real counterparts' behavior
- Add support for datetime/date
- Add support for decimal/cdecimal
- Add support for mapping from uncompressed zip files
- Treat long as int
- More informative packing error messages
- Add pool module with dynamic object pool implementations, allowing
  incremental build of large object heirarchies/collections.
- Add GenericFileMapper utility class to get buffers out of files
- Add iter() and iter_fast() methods that allow optimized iteration
  through proxied_list s containing objects, by allowing proxy
  reuse and masked iteration. The first one can take a proxy to use,
  while the second one will build its own generic proxy.
- Use fused types to reduce code bloat and support more integer formats.

Changes
-------

- Allow building with Cython 0.28 and above
- Use `v` prefix on releases to have fixed links for this document
- mapped_list now returns actual lists and not a subclass
- Use a strong-referencing id map by default, making it safer for cases
  with nonstandard or unmanaged object lifetimes
- Support packing proxies as if they were the original thing in most
  cases. Nested uses require schema registration. This allows constructing
  shared buffers out of other shared buffers.
- Improved idmap handling for the case of repacking proxies. It still may
  fail to recognize primitive object identity properly since proxies will
  return a unique object on each access, inflating the resulting buffer
  perhaps considerably. Proper identity was implemented for proxied
  containers though.
- Shrink some buffers by employing narrow pointers where possible
- Optimized sequence packing
- New wide bitmap frozenset format allows more frozensets to be packed
  as bitmaps.
- Offsets are Py_ssize_t now. That shouldn't be a noticeable change,
  unless you've got more storage than the universe.
- Improved performance of binary search utilities.
- NumericId32[Multi]Mapper and StringId32[Multi]Mapper are now built-in
  classes when cythonized (should be relatively transparent).
- Schema.pack is now thread-safe if the module is cythonized


Bugfixes
--------

- Fix a buffer reference leak in proxies when building with Cython

v0.4.8 - 2018-05-28
===================

Bugfixes
--------

- Reduce memory usage of MHM index merge, it was unnecessarily
  holding onto temporary intermediate results

v0.4.7 - 2018-02-22
===================

Bugfixes
--------

- Solve issue when using lz4 0.18.1

v0.4.6 - 2017-10-18
===================

Bugfixes
--------

- Reject Cython >= 0.27.1, since they break the build

v0.4.5 - 2017-10-12
===================

Changes
-------

- Small performance optimizations

v0.4.4 - 2017-10-02
===================

Bugfixes
--------

- Fix on-disk IdMapper build which in some cases could build
  an unsorted (ie: broken) MHM.

v0.4.3 - 2017-09-28
===================

Changes
-------

- Unpack frozensets as actual frozensets, not subclasses of it
  (mapped_frozenset). Matches the behavior of other primitve
  unpackers and allows singleton empty sets to be unpacked as
  singletons
- Significantly reduce memory usage during IdMapper builds,
  especially when given a tempdir where to stage temprary data

v0.4.2 - 2017-08-14
===================

Bugfixes
--------

- Fix integer overflow in handling masks that made a subtle mess of
  things if you had more than 32 attributes

Changes
-------

- Reduce peak memory usage during IdMapper builds, especially when
  using deduplication

v0.4.1 - 2017-07-18
===================

Bugfixes
--------

- Fix complex schema unpickling. When schemas contained typed object
  references with their own schema, unpickling wouldn't properly
  register the unpickled schemas with the mapped_object proxy
  factory, and would fail to build the required objects with a KeyError

v0.4.0 - 2017-07-12
===================

Bugfixes
--------

- Several fixes to pure-python mode
- Ensure deterministic attribute ordering when using dict slot_types

Added
-----

- Schema objects are now picklable, and pickled schemas preserve attribute
  ordering, so they can be used to safely unpack objects packed by external
  code
- MappedArrayProxyBase now embeds the schema used during build, so they can
  be safely mapped from other interpreters and versions of the code, as long
  as client code still understands the foreign schemas. That is, as long as
  schemas are source-code compatible
- Fix setup.py to fail properly when built without Cython and without an explicit
  disable of cython optimizations
- Fix setup.py to try to automatically install Cython as a dependency if not present
- Require setuptools 20.0 and above. Earlier versions don't interact well with Cython
- Added ability to efficiently merge numeric and approximate id mappers
  (not yet supported for exact mappers)

v0.3.3 - 2017-04-25
===================

Bugfixes
--------

- Limit pack buffer expansion to avoid memory exhaustion on recurring errors
- Fix bitmap generation for objects with more than 30 attributes

v0.3.2 - 2017-04-07
===================

Bugfixes
--------

- Fix unpacking of frozensets with big (beyond 32-bit) offsets

v0.3.1 - 2016-11-09
===================

Bugfixes
--------

- Fix binary search functions to properly test the given array's dtype to avoid
  spurious NotImplementedError s

v0.3.0 - 2016-11-08
===================

Added
-----

- Exported hinted_bsearch and bsearch functions (present when cythonized) that implement
  both hinted and regular binary search on numpy arrays. Unlike numpy's bsearch, they don't
  release the GIL, so they're faster for single lookups than numpy's counterparts
- Added sorted_contains and hinted_sorted_contains as useful helpers to use sorted numpy
  arrays as compact number sets

v0.2.1 - 2016-10-18
===================

Bugfixes
--------

- Make requirements install requirements
- Add __version__
- Fix pure-python compatibility
- Implement more of the mapping interface on id mappers
- Add get_iter and __contains__ on multimappers that avoids materializing big sequences

.. _0.2.0:

v0.2.0 - 2016-10-11
===================

Bugfixes
--------

- Fix requirements to include chorde_

Added
-----

- Add multimaps, binary compatible with simple mappings,
  but return all matching values for a key rather than a single one
- Add approximate string multimaps

.. _0.1.1:

v0.1.0
======

*Note*: this release has dependency issues, use 0.2.0_ instead

Added
-----

- Initial release

.. _chorde: https://bitbucket.org/claudiofreire/chorde
.. _keepachangelog: http://keepachangelog.com/

.. _BTS-90: https://jampphq.atlassian.net/browse/BTS-90
.. _BTS-619: https://jampphq.atlassian.net/browse/BTS-619
.. _BTS-618: https://jampphq.atlassian.net/browse/BTS-618
