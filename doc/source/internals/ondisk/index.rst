On-Disk format
==============

This section will describe the on-buffer format of data bundles created with sharedbuffers.

Since buffers are usually memory-mapped files, this also describes the on-disk format.

We'll define all data structures in a C-struct-like language. When the semantic differs from what a C compiler
might interpret, a clarification will be made.

All integers larger than a byte are stored in native endian format. This means buffers aren't compatible
across platforms with different endianness. As such, usage of shared buffers in network protocol messages
should be done with care. See the :ref:`schema-portability` section.

.. toctree::
    :glob:

    *
