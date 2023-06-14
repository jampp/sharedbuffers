# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tempfile
import mmap
import struct

from .mapped_struct import StrongIdMap
import six
if six.PY3:
    buffer = memoryview


# Default section size is set to 128MB which is a size at which most
# malloc implementations turn to mmap
DEFAULT_SECTION_SIZE = 128<<20

DEFAULT_PACK_BUFFER = 65536

class Section(object):

    def __init__(self, buf, implicit_offs=0, idmap_kwargs={}):
        self.buf = buf
        self.real_buf = buffer(buf)
        self.attach(implicit_offs, idmap_kwargs)

    def allocate(self, size=None):
        write_pos = self.write_pos
        if size is None:
            size = len(self.buf) - write_pos
        if (write_pos + size) <= len(self.buf):
            self.write_pos += size
            return write_pos
        else:
            raise IndexError("Buffer overflow trying to allocate %d bytes from section" % size)

    def append(self, buf, verify_pos=None):
        write_pos = self.allocate(len(buf))
        if verify_pos is not None and verify_pos != write_pos:
            raise RuntimeError("Concurrent modification")
        self.buf[write_pos:write_pos+len(buf)] = bytes(buf)
        return write_pos

    @property
    def free_space(self):
        return len(self.buf) - self.write_pos

    def detach(self):
        del self.idmap

    def attach(self, implicit_offs=0, idmap_kwargs={}):
        self.implicit_offs = implicit_offs
        self.write_pos = 0
        self.idmap = StrongIdMap(**idmap_kwargs)

    def __del__(self):
        buf = self.buf
        self.real_buf = None
        self.buf = None
        if hasattr(buf, 'close'):
            buf.close()

class BaseObjectPool(object):
    """
    Base abstract class for object pools. Use a concrete implementation instead.

    An object pool can be dynamically grown by appending object data one object
    at a time.

    The object pool is split into large-ish sections that are memory-mapped files.
    Big data structures can be built onto them without requiring equal amounts of
    memory by handling object proxies instead of actual objects.

    Each section has its own :term:`idmap`, which means object deduplication cannot
    work across sections, so section size has to be balanced to be large, but not
    too large, since the pool grows in whole-section increments.

    Object pools can be useful to parallelize the construction of big data structures.
    Relocatable object data can be packed in a process pool, and returned as a byte
    string that can then be added to the object pool. This construction method isn't
    as effective at reference deduplication, since relocatable object data needs to
    be closed (have no external references), but in many cases the bulk of big data
    structures is composed of easy to close objects.

    They can also help in building large object graphs that don't fit in memory, by
    replacing heavy objects with lightweight proxies as they are built and pushed
    into the pool one object at a time.
    """

    def _mktemp(self):
        raise NotImplementedError

    def __init__(self, section_size=DEFAULT_SECTION_SIZE, temp_kwargs={}, idmap_kwargs={},
            section_freelist=None):
        """
        :param int section_size: The size of each section in the pool

        :param dict temp_kwargs: Keywords passed to :class:`tempfile.TemporaryFile` to
            customize tempfile allocation.

        :param dict idmap_kwargs: Keywords passed when constructing new :class:`.StrongIdMap`
            instances. Particularly useful to pass a stable set of known stable objects
            to improve reference deduplication.

        :param list section_freelist: *(optional)* If given, a list to hold section freed
            when :meth:`close` is invoked. If multiple pools are constructed in
            succession, this can speed up the process of allocating new sections
            by reusing discarded sections.
        """
        self.temp_kwargs = temp_kwargs
        self.idmap_kwargs = idmap_kwargs
        self.section_size = section_size
        self.sections = []
        self.total_size = 0
        self.idmap_preload = []
        self.section_freelist = section_freelist

    def __del__(self):
        self.idmap_kwargs = None
        self.temp_kwargs = None
        self.close(True)

    @property
    def size(self):
        """
        Total size in bytes of all data appended to this pool.
        """
        if self.sections:
            last_section = self.sections[-1]
            return last_section.implicit_offs + last_section.write_pos
        else:
            return 0

    def add_section(self):
        """
        Append a new section to the pool, and return it.
        """
        f = None
        try:
            new_section = None
            implicit_offs = self.total_size

            if self.section_freelist:
                new_section = self.section_freelist.pop()
                new_section.attach(implicit_offs, self.idmap_kwargs)

            if new_section is None:
                f = self._mktemp()
                f.truncate(self.section_size)
                buf = mmap.mmap(
                    f.fileno(), 0,
                    flags = mmap.MAP_SHARED,
                    access = mmap.ACCESS_WRITE)
                new_section = Section(buf, implicit_offs, self.idmap_kwargs)
                new_section.fileobj = f
            else:
                buf = new_section.buf

            self.total_size += len(buf)

            # Initialize with preloaded items
            if self.idmap_preload:
                idmap = {}
                try:
                    for schema, obj in self.idmap_preload:
                        self._pack_into(schema, obj, new_section, None, idmap)
                except (struct.error, IndexError):
                    raise RuntimeError("Preload items dont't fit in empty section, increase section size")
                new_section.idmap.preload(idmap)
        except:
            if f is not None:
                f.close()
            raise

        self.sections.append(new_section)
        return new_section

    def _pack_into(self, schema, obj, section, min_pack_buffer_cell=None, idmap=None, pack_buffer=None):
        write_pos = section.write_pos
        if idmap is None:
            idmap = section.idmap
        implicit_offs = section.implicit_offs + write_pos
        if hasattr(schema, 'pack'):
            buf = schema.pack(obj, idmap, implicit_offs=implicit_offs)
        elif hasattr(schema, 'pack_into'):
            if pack_buffer is None:
                if min_pack_buffer_cell:
                    pack_buffer_size = min_pack_buffer_cell[0]
                else:
                    pack_buffer_size = 1 << 20
                pack_buffer = bytearray(pack_buffer_size)
            endp = schema.pack_into(obj, pack_buffer, 0, idmap, implicit_offs=implicit_offs)
            buf = pack_buffer[:endp]
        if min_pack_buffer_cell:
            min_pack_buffer_cell[0] = max(min_pack_buffer_cell[0], len(buf))
        return section.append(buf, write_pos)

    def pack(self, schema, obj, min_pack_buffer=DEFAULT_PACK_BUFFER, clear_idmaps_on_new_section=True,
            pack_buffer=None):
        """
        Add an object to the pool, and return a proxy to it.

        :param Schema schema: The :class:`~sharedbuffers.mapped_struct.Schema` of the object data being pushed

        :param obj: Object to be packed into the pool.

        :param int min_pack_buffer: *(optional)* Minimum required free space in the section. If the section contains
            less than this amount of free space, a new empty section is allocated without even trying
            to pack the object in the almost-full section.

        :param bool clear_idmaps_on_new_section: *(default True)* Whether to clear the :term:`idmap` s of full sections
            when new sections are allocated. Doing this can keep memory usage low, but prevent efficient reuse of free
            section space. The default is usually ok.

        :param bytearray pack_buffer: *(optional)* A buffer used to build the object data before copying it into the
            pool. If none is provided, one is allocated automatically.

        :rtype: tuple[int, proxy]
        :returns: A pair with the location of the added object and a proxy to the object itself.
        """
        sections = self.sections
        _min_pack_buffer=[min_pack_buffer]
        for section in reversed(sections):
            if section.free_space < _min_pack_buffer[0]:
                continue

            try:
                pos = self._pack_into(schema, obj, section, _min_pack_buffer)
            except (struct.error, IndexError):
                # Possibly corrupt
                section.idmap.clear()
            else:
                break
        else:
            if clear_idmaps_on_new_section:
                self.clear_idmaps()
            section = self.add_section()
            pos = self._pack_into(schema, obj, section)
        return pos + section.implicit_offs, schema.unpack_from(section.real_buf, pos)

    def add(self, schema, buf, clear_idmaps_on_new_section=True):
        """
        Add object data to the pool, and return a proxy to it.

        Make sure the contents of ``buf`` are relocatable (ie: have no external references)

        :param Schema schema: The :class:`~sharedbuffers.mapped_struct.Schema` of the object data being pushed

        :param buffer buf: Object data produced with :meth:`sharedbuffers.mapped_struct.Schema.pack_into`
            or a similar method.

        :rtype: tuple[int, proxy]
        :returns: A pair with the location of the added object and a proxy to the object itself.
        """
        sections = self.sections
        for section in sections:
            if section.free_space < len(buf):
                continue

            try:
                pos = section.append(buf, section.write_pos)
            except (struct.error, IndexError):
                pass
            else:
                break
        else:
            if clear_idmaps_on_new_section:
                self.clear_idmaps()
            section = self.add_section()
            pos = section.append(buf, section.write_pos)
        return pos + section.implicit_offs, schema.unpack_from(section.real_buf, pos)

    def add_preload(self, schema, obj):
        """
        Preload this object and all its contents into all the individual sections.
        Fix those in the per-section idmap. It can improve both speed and reduce size
        if the objects to be appended references the objects contained in the preloaded
        objects a lot, but the objects will be repeated on each section, so they should
        be small or their size overhead will outweight their benefit.

        :param Schema schema: The :class:`~sharedbuffers.mapped_struct.Schema` describing the object's shape

        :param obj: The object to be preloaded. It will automatically be packed each time
            a new section is added.
        """
        self.idmap_preload.append((schema, obj))

    def find_section(self, pos):
        """
        :rtype: Section
        :returns: The section where the logical position ``pos`` resides.
        """
        for section in self.sections:
            if section.implicit_offs <= pos < section.implicit_offs + len(section.buf):
                return section

    def unpack(self, schema, pos):
        """
        Unpacks object data from the logical position ``pos`` using the given
        :class:`~sharedbuffers.mapped_struct.Schema`.

        :param Schema schema: The expected schema of the object at ``pos``.

        :param int pos: The logical position from which to unpack the object.

        :return: A proxy to the object.
        """
        section = self.find_section(pos)
        if section is None:
            raise IndexError("Position %d out of bounds for object pool" % pos)
        return schema.unpack_from(section.real_buf, pos - section.implicit_offs)

    def clear_idmaps(self):
        """
        Clears all :term:`idmap` s in all sections.
        """
        for section in self.sections:
            section.idmap.clear()

    def dump(self, fileobj):
        """
        Dump the whole pool's content into ``fileobj``.

        :type fileobj: file-like
        :param fileobj: The file or file-like object onto which to dump all data.

        :rtype: int
        :returns: The amount of data written
        """
        if not self.sections:
            return 0

        for section in self.sections[:-1]:
            fileobj.write(section.real_buf)
        section = self.sections[-1]

        # Align to 8-bytes
        write_bytes = section.write_pos
        write_bytes += (8 - write_bytes % 8) % 8
        fileobj.write(section.real_buf[:write_bytes])
        return section.implicit_offs + write_bytes

    def close(self, discard=False):
        """
        Release all pool resources. Resets the pool to empty state.

        :type discard: bool
        :param discard: If True, it will close all underlying files instead of
            releasing them into the section freelist for reuse. Use close(True)
            when immediate resource release is needed. If files are temporary
            files, their contents will be lost (cannot be reopened).
        """
        sections = self.sections
        self.sections = []

        if self.section_freelist is not None:
            for section in sections:
                section.detach()
            if discard:
                sections.extend(self.section_freelist)
                del self.section_freelist[:]
            else:
                self.section_freelist.extend(sections)
                del sections[:]

        for section in sections:
            fileobj = getattr(section, 'fileobj', None)
            if fileobj is not None:
                fileobj.close()

        self.total_size = 0

class TemporaryObjectPool(BaseObjectPool):
    """
    Implementation of :class:`BaseObjectPool` using anonymous temporary files.
    """

    def _mktemp(self):
        return tempfile.TemporaryFile(**self.temp_kwargs)
