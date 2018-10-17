# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tempfile
import mmap
import struct

# Default section size is set to 128MB which is a size at which most
# malloc implementations turn to mmap
DEFAULT_SECTION_SIZE = 128<<20

DEFAULT_PACK_BUFFER = 65536

class Section(object):

    def __init__(self, buf, implicit_offs=0):
        self.buf = buf
        self.real_buf = buffer(buf)
        self.implicit_offs = implicit_offs
        self.write_pos = 0
        self.idmap = {}

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

    def __init__(self, section_size=DEFAULT_SECTION_SIZE, temp_kwargs={}):
        self.temp_kwargs = temp_kwargs
        self.section_size = section_size
        self.sections = []
        self.freehead = 0
        self.total_size = 0

    def add_section(self):
        with self._mktemp() as f:
            f.truncate(self.section_size)
            buf = mmap.mmap(
                f.fileno(), 0,
                flags = mmap.MAP_SHARED,
                prot = mmap.PROT_READ|mmap.PROT_WRITE,
                access = mmap.ACCESS_READ|mmap.ACCESS_WRITE)

        implicit_offs = self.total_size
        self.total_size += len(buf)
        new_section = Section(buf, implicit_offs)
        self.sections.append(new_section)
        return new_section

    def _pack_into(self, schema, obj, section, min_pack_buffer_cell=None):
        write_pos = section.write_pos
        buf = schema.pack(obj, implicit_offs=section.implicit_offs + write_pos)
        if min_pack_buffer_cell:
            min_pack_buffer_cell[0] = max(min_pack_buffer_cell[0], len(buf))
        return section.append(buf, write_pos)

    def pack(self, schema, obj, min_pack_buffer=DEFAULT_PACK_BUFFER):
        sections = self.sections
        _min_pack_buffer=[min_pack_buffer]
        for i in xrange(self.freehead, len(sections)):
            section = sections[i]
            if section.free_space < _min_pack_buffer[0]:
                continue

            try:
                pos = self._pack_into(schema, obj, section, _min_pack_buffer)
            except (struct.error, IndexError):
                pass
            else:
                break
        else:
            section = self.add_section()
            pos = self._pack_into(schema, obj, section)
        return pos, schema.unpack_from(section.real_buf, pos)

    def clear_idmaps(self):
        for section in self.sections:
            self.idmap.clear()

class TemporaryObjectPool(BaseObjectPool):

    def _mktemp(self):
        return tempfile.TemporaryFile(**self.temp_kwargs)
