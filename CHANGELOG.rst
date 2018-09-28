Change Log
==========

All notable changes to this project will be documented here.

The format is largely inspired by keepachangelog_.

.. _0.1.1:

v0.5.0 - Unreleased
===================

Added
-----

- Add support for proxied buffers
- Add support for proxied numpy arrays
- Add support for proxied list/tuple
- Add support for mapped dict
- Allow specifying custom bases for proxy objects, so proxies
  can inherit their real counterparts' behavior
- Add support for datetime/date
- Add support for decimal/cdecimal
- Add support for mapping from uncompressed zip files
- Treat long as int
- More informative packing error messages

Changes
-------

- Allow building with Cython 0.28 and above
- Use `v` prefix on releases to have fixed links for this document

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

v0.1.0
=====

*Note*: this release has dependency issues, use 0.2.0_ instead

Added
-----

- Initial release

.. _chorde: https://bitbucket.org/claudiofreire/chorde
.. _keepachangelog: http://keepachangelog.com/

