# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tempfile
import mmap
import struct

from .mapped_struct import StrongIdMap

# Default section size is set to 128MB which is a size at which most
# malloc implementations turn to mmap
DEFAULT_SECTION_SIZE = 128<<20

DEFAULT_PACK_BUFFER = 65536

class Section(object):

    def __init__(self, buf, implicit_offs=0, idmap_kwargs={}):
        self.buf = buf
        self.real_buf = buffer(buf)
        self.implicit_offs = implicit_offs
        self.write_pos = 0
        self.idmap = StrongIdMap(**idmap_kwargs)

    def allocate(self, size=None):
        write_pos = self.write_pos
        if size is None:
            size = len(self.buf) - write_pos
        if (write_pos + size) <= len(self.buf):
            self.write_pos += size
            return write_pos
        else:
            raise IndexError("Buffer overflow trying to allocate %d bytes from section" % size)

    def resize(self, pos, new_size):
        new_end = pos + new_size
        if new_end != self.write_pos:
            raise RuntimeError("Cannot resize non-tail buffers")
        self.write_pos = new_end

    def append(self, buf, verify_pos=None):
        write_pos = self.allocate(len(buf))
        if verify_pos is not None and verify_pos != write_pos:
            raise RuntimeError("Concurrent modification")
        self.buf[write_pos:write_pos+len(buf)] = bytes(buf)
        return write_pos

    @property
    def free_space(self):
        return len(self.buf) - self.write_pos

class BaseObjectPool(object):

    def _mktemp(self):
        raise NotImplementedError

    def __init__(self, section_size=DEFAULT_SECTION_SIZE, temp_kwargs={}, idmap_kwargs={}):
        self.temp_kwargs = temp_kwargs
        self.idmap_kwargs = idmap_kwargs
        self.section_size = section_size
        self.sections = []
        self.freehead = 0
        self.total_size = 0
        self.idmap_preload = []

    @property
    def size(self):
        if self.sections:
            last_section = self.sections[-1]
            return last_section.implicit_offs + last_section.write_pos
        else:
            return 0

    def add_section(self):
        f = self._mktemp()
        try:
            f.truncate(self.section_size)
            buf = mmap.mmap(
                f.fileno(), 0,
                access = mmap.ACCESS_WRITE)

            implicit_offs = self.total_size
            self.total_size += len(buf)
            new_section = Section(buf, implicit_offs, self.idmap_kwargs)
            new_section.fileobj = f

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
            f.close()
            raise

        self.sections.append(new_section)
        return new_section

    def _pack_into(self, schema, obj, section, min_pack_buffer_cell=None, idmap=None):
        write_pos = section.write_pos
        if idmap is None:
            idmap = section.idmap
        buf = schema.pack(obj, idmap, implicit_offs=section.implicit_offs + write_pos)
        if min_pack_buffer_cell:
            min_pack_buffer_cell[0] = max(min_pack_buffer_cell[0], len(buf))
        return section.append(buf, write_pos)

    def pack(self, schema, obj, min_pack_buffer=DEFAULT_PACK_BUFFER, clear_idmaps_on_new_section=True):
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

    def add_preload(self, schema, obj):
        """
        Preload this object and all its contents into all the individual sections.
        Fix those in the per-section idmap. Can improve both speed and reduce size
        if the objects to be appended references the objects contained in the preloaded
        objects a lot.
        """
        self.idmap_preload.append((schema, obj))

    def find_section(self, pos):
        for section in self.sections:
            if section.implicit_offs <= pos < section.implicit_offs + len(section.buf):
                return section

    def unpack(self, schema, pos):
        section = self.find_section(pos)
        if section is None:
            raise IndexError("Position %d out of bounds for object pool" % pos)
        return schema.unpack_from(section.real_buf, pos - section.implicit_offs)

    def clear_idmaps(self):
        for section in self.sections:
            section.idmap.clear()

    def dump(self, fileobj):
        if not self.sections:
            return 0

        for section in self.sections[:-1]:
            write_bytes = len(section.real_buf)
            fileobj.write(section.real_buf)
        section = self.sections[-1]

        # Align to 8-bytes
        write_bytes = section.write_pos
        write_bytes += (8 - write_bytes % 8) % 8
        fileobj.write(section.real_buf[:write_bytes])
        return section.implicit_offs + write_bytes

class TemporaryObjectPool(BaseObjectPool):

    def _mktemp(self):
        return tempfile.NamedTemporaryFile(**self.temp_kwargs)
