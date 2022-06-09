# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
from .mapped_struct import TestStruct

from sharedbuffers import mapped_struct, pool
from six import iteritems
from six.moves import xrange

class ContainerStruct(TestStruct):
    __slot_types__ = {
        'a' : int,
        'b' : float,
        'fset' : frozenset,
        't' : tuple,
        'l' : list,
        'pt' : mapped_struct.proxied_tuple,
        'pl' : mapped_struct.proxied_list,
        'o' : object
    }

class SmallIntContainerPackingTest(unittest.TestCase):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'a' : 20,
        'b' : 3.24,
        'fset' : frozenset([1,3,7]),
        't' : (3,6,7),
        'l' : [1,2,3],
        'pt' : (3,6,7),
        'pl' : [1,2,3],
        'o' : [{"a": (3,4,5), "b": frozenset([1000,3444,525])}],
    }]

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)

    def testPack(self, schema = None):
        if schema is None:
            schema = self.schema
        p = pool.TemporaryObjectPool()
        self.addCleanup(p.close, True)
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            dx = p.pack(schema, x)[1]
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testPackObject(self):
        # hack - unregister schema
        mapped_struct.mapped_object.TYPE_CODES.pop(self.Struct, None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop(b'\xfe', None)
        mapped_struct.mapped_object.register_schema(self.Struct, self.schema, b'\xfe')
        self.testPack(schema=mapped_struct.mapped_object)

    def testPreload(self):
        p = pool.TemporaryObjectPool()
        self.addCleanup(p.close, True)
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            p.add_preload(self.schema, x)
        for TEST_VALUES in self.TEST_VALUES:
            dx = p.pack(self.schema, x)[1]
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testOverflow(self):
        p = pool.TemporaryObjectPool(section_size=4096)
        self.addCleanup(p.close, True)
        for i in xrange(300):
            for TEST_VALUES in self.TEST_VALUES:
                x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
                dx = p.pack(self.schema, x)[1]
                for k,v in iteritems(TEST_VALUES):
                    self.assertTrue(hasattr(dx, k))
                    self.assertEqual(getattr(dx, k), v)
                for k in self.Struct.__slots__:
                    if k not in TEST_VALUES:
                        self.assertFalse(hasattr(dx, k))
        self.assertGreater(len(p.sections), 1)

    def testStableSet(self):
        temp_idmap = {}
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            self.schema.pack(x, idmap=temp_idmap)

        p = pool.TemporaryObjectPool(
            section_size=1<<20,
            idmap_kwargs=dict(stable_set=set(temp_idmap)))
        self.addCleanup(p.close, True)
        del temp_idmap

        for i in xrange(300):
            for TEST_VALUES in self.TEST_VALUES:
                x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
                dx = p.pack(self.schema, x)[1]
                for k,v in iteritems(TEST_VALUES):
                    self.assertTrue(hasattr(dx, k))
                    self.assertEqual(getattr(dx, k), v)
                for k in self.Struct.__slots__:
                    if k not in TEST_VALUES:
                        self.assertFalse(hasattr(dx, k))

    def testUnpack(self):
        p = pool.TemporaryObjectPool()
        self.addCleanup(p.close, True)
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            pos = p.pack(self.schema, x)[0]
            dx = p.unpack(self.schema, pos)
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))
