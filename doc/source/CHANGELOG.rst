Change Log
..........

All notable changes to this project will be documented here.

The format is largely inspired by keepachangelog_.

v0.8.2 - Unreleased
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

