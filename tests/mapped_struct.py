# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
import itertools
import operator
import tempfile
import os

from sharedbuffers import mapped_struct

class SimpleStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.iteritems():
            setattr(self, k, v)

class SizedNumericStruct(object):
    __slot_types__ = {
        'a' : mapped_struct.int32,
        'b' : mapped_struct.float32,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.iteritems():
            setattr(self, k, v)

class PrimitiveStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
        's' : str,
        'u' : unicode,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.iteritems():
            setattr(self, k, v)

class ContainerStruct(object):
    __slot_types__ = {
        'fset' : frozenset,
        't' : tuple,
        'l' : list,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.iteritems():
            setattr(self, k, v)

class ObjectStruct(object):
    __slot_types__ = {
        'o' : object,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.iteritems():
            setattr(self, k, v)

class BasePackingTestMixin(object):
    Struct = None
    
    TEST_VALUES = [{}]

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
    
    def testAllUnset(self):
        x = self.Struct()
        dx = self.schema.unpack(self.schema.pack(x))
        for k in self.Struct.__slots__:
            self.assertFalse(hasattr(dx, k))

    def testAllNone(self):
        x = self.Struct(**{k:None for k in self.Struct.__slots__})
        dx = self.schema.unpack(self.schema.pack(x))
        for k in self.Struct.__slots__:
            self.assertTrue(hasattr(dx, k))
            self.assertIsNone(getattr(dx, k))

    def testPackUnpack(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in TEST_VALUES.iteritems()})
            dx = self.schema.unpack(self.schema.pack(x))
            for k,v in TEST_VALUES.iteritems():
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testUnpackInto(self):
        dx = self.schema.Proxy()
        for TEST_VALUES in self.TEST_VALUES * 2:
            x = self.Struct(**{k:v for k,v in TEST_VALUES.iteritems()})
            px = self.schema.pack(x)
            dx = self.schema.unpack(px, proxy_into = dx)
            for k,v in TEST_VALUES.iteritems():
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

class SimplePackingTest(BasePackingTestMixin, unittest.TestCase):
    Struct = SimpleStruct
    TEST_VALUES = [
        {
            'a' : 3,
            'b' : 4.0,
        },
        {
            'b' : 4.0,
        },
        {
            'a' : 3,
            'b' : None,
        },
    ]

class SizedNumericPackingTest(SimplePackingTest):
    Struct = SizedNumericStruct

class PrimitivePackingTest(SimplePackingTest):
    Struct = PrimitiveStruct
    TEST_VALUES = [
        {
            'a' : 3,
            'b' : 4.0,
            's' : b'HOLA\x00\x01',
            'u' : u'HÓLá€Á',
        },
        {
            'a' : 3,
            'b' : 4.0,
            'u' : None,
        },
        {
            'a' : 3,
            'b' : 4.0,
            's' : b'',
            'u' : u'',
        },
        {
            'a' : 3,
            'b' : 4.0,
            's' : b'\x00fkal' * 300,
            'u' : u'Höø3lkßba€' * 100,
        },
        {
            'a' : 3,
            'b' : 4.0,
            's' : os.urandom(256000),
            'u' : os.urandom(256000).decode("latin1"),
        },
    ]

class SmallIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1,3,7]),
        't' : (3,6,7),
        'l' : [1,2,3],
    }]

class ShortIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1000,3000,7000]),
        't' : (3000,6000,7000),
        'l' : [1000,2000,3000],
    }]

class LongIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([0x12341234,0x23452345,0x43534353]),
        't' : (0x12241234,0x23352345,0x43334353),
        'l' : [0x13341234,0x24452345,0x45534353],
    }]

class LongLongIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([0xf012341234,0xf023452345,0xf043534353]),
        't' : (0xf012241234,0xf023352345,0xf043334353),
        'l' : [0xf013341234,0xf024452345,0xf045534353],
    }]

class FloatContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1.0,3.0,7.0]),
        't' : (3.0,6.0,7.0),
        'l' : [1.0,2.0,3.0],
    }]

class BytesContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset(['1.0','3.0','7.0']),
        't' : ('3.0','6.0','7.0'),
        'l' : ['1.0','2.0','3.0'],
    }]

class NestedContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([(1,2),(3,4),(6,7)]),
        't' : ((3,),(6,7,8),(1,7)),
        'l' : [[1],[2,3],(3,4)],
    }]

class ObjectPackagingTest(SimplePackingTest):
    Struct = ObjectStruct
    TEST_VALUES = [
        { 'o' : 1 },
        { 'o' : True },
        { 'o' : 3.0 },
        { 'o' : [ 1,2,3 ] },
        { 'o' : ( 2,3,4 ) },
        { 'o' : frozenset([ 1,3,8 ]) },
        { 'o' : "blabla" },
        { 'o' : u"bláblá€" },
    ]

class NestedObjectPackagingTest(SimplePackingTest):
    Struct = ObjectStruct
    SubStruct = ContainerStruct
    subschema = mapped_struct.Schema.from_typed_slots(SubStruct)
    
    TEST_VALUES = [
        { 
            'o' : ContainerStruct(**{
                'fset' : frozenset([1000,3000,7000]),
                't' : (3000,6000,7000),
                'l' : [1000,2000,3000],
            }),
        },
    ]

    def setUp(self):
        mapped_struct.mapped_object.register_schema(self.SubStruct, self.subschema, '{')
        super(NestedObjectPackagingTest, self).setUp()

    def assertEqual(self, value, expected, *p, **kw):
        if isinstance(expected, self.SubStruct):
            for k in self.SubStruct.__slots__:
                self.assertEqual(hasattr(value, k), hasattr(expected, k), *p, **kw)
                if hasattr(expected, k):
                    self.assertEqual(getattr(value, k), getattr(expected, k), *p, **kw)
        else:
            return super(NestedObjectPackagingTest, self).assertEqual(value, expected, *p, **kw)

class MappedArrayTest(unittest.TestCase):
    Struct = ContainerStruct
    TEST_VALUES = [
        {
            'fset' : frozenset(['1.0','3.0','7.0']),
            't' : ('3.0','6.0','7.0'),
            'l' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset([1,3,7]),
            't' : (3,6,7),
            'l' : [1,2,3],
        }, {
            'fset' : frozenset([1000,3000,7000]),
            't' : (3000,6000,7000),
            'l' : [1000,2000,3000],
        }, {
            'fset' : frozenset([0x12341234,0x23452345,0x43534353]),
            't' : (0x12241234,0x23352345,0x43334353),
            'l' : [0x13341234,0x24452345,0x45534353],
        }, {
            'fset' : frozenset([0xf012341234,0xf023452345,0xf043534353]),
            't' : (0xf012241234,0xf023352345,0xf043334353),
            'l' : [0xf013341234,0xf024452345,0xf045534353],
        }, {
            'fset' : frozenset([1.0,3.0,7.0]),
            't' : (3.0,6.0,7.0),
            'l' : [1.0,2.0,3.0],
        }, {
            'fset' : frozenset(['1.0','3.0','7.0']),
            't' : ('3.0','6.0','7.0'),
            'l' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset(['1.0',os.urandom(128000),'7.0']),
            't' : ('3.0',os.urandom(256000),'7.0'),
            'l' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset([(1,2),(3,4),(6,7)]),
            't' : ((3,),(6,7,8),(1,7)),
            'l' : [[1],[2,3],(3,4)],
        }
    ]

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
        class MappedArrayClass(mapped_struct.MappedArrayProxyBase):
            schema = self.schema
        self.MappedArrayClass = MappedArrayClass
        self.test_values = [ self.Struct(**kw) for kw in self.TEST_VALUES ]

    def _checkValues(self, mapping, iterator):
        self.assertEqual(len(self.test_values), len(mapping))
        for reference, proxy in itertools.izip(self.test_values, iterator):
            self.assertEqual(reference.fset, proxy.fset)
            self.assertEqual(reference.t, proxy.t)
            self.assertEqual(reference.l, proxy.l)

    def testBuildNoIdmap(self):
        self.MappedArrayClass.build(self.test_values)

    def testBuildWithIdmap(self):
        self.MappedArrayClass.build(self.test_values, idmap = {})

    def testBuildEmpty(self):
        self.MappedArrayClass.build([])

    def testFileMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedArrayClass.map_file(destfile)
            self._checkValues(mapped, mapped)

    def testBufferMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            destfile.seek(0)
            mapped = self.MappedArrayClass.map_buffer(buffer(destfile.read()))
            self._checkValues(mapped, mapped)

    def testFastIteration(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedArrayClass.map_file(destfile)
            self._checkValues(mapped, mapped.iter_fast())

class MappedMappingTest(unittest.TestCase):
    # Reuse test values from MappedArrayTest to simplify test code
    Struct = MappedArrayTest.Struct
    TEST_VALUES = MappedArrayTest.TEST_VALUES
    TEST_KEYS = [ x*7 for x in xrange(len(TEST_VALUES)) ]
    IdMapperClass = mapped_struct.NumericIdMapper

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
        class MappedArrayClass(mapped_struct.MappedArrayProxyBase):
            schema = self.schema
        class MappedMappingClass(mapped_struct.MappedMappingProxyBase):
            ValueArray = MappedArrayClass
            IdMapper = self.IdMapperClass
        self.MappedMappingClass = MappedMappingClass
        self.test_values = {
            k : self.Struct(**kw)
            for k,kw in zip(self.TEST_KEYS, self.TEST_VALUES)
        }

    def assertStructEquals(self, reference, proxy):
        self.assertEqual(reference.fset, proxy.fset)
        self.assertEqual(reference.t, proxy.t)
        self.assertEqual(reference.l, proxy.l)

    def checkStructEquals(self, reference, proxy):
        return (
            (reference.fset == proxy.fset)
            and (reference.t == proxy.t)
            and (reference.l == proxy.l)
        )

    def assertMappingOk(self, mapping, test_values = None):
        if test_values is None:
            test_values = self.test_values

        # test basic attributes
        self.assertEqual(len(test_values), len(mapping))

        # test key iteration and enumeration
        self.assertEqual(set(test_values.keys()), set(mapping.keys()))
        self.assertEqual(set(test_values.iterkeys()), set(mapping.iterkeys()))

        # test lookup
        for k,reference in test_values.iteritems():
            self.assertStructEquals(reference, mapping.get(k))
        for k,reference in test_values.iteritems():
            self.assertStructEquals(reference, mapping[k])

        # test item iteration and enumeration
        for k,proxy in mapping.iteritems():
            reference = self.test_values[k]
            self.assertStructEquals(reference, proxy)
        for k,proxy in mapping.items():
            reference = self.test_values[k]
            self.assertStructEquals(reference, proxy)

    def testBuildNoIdmap(self):
        self.MappedMappingClass.build(self.test_values)

    def testBuildWithIdmap(self):
        self.MappedMappingClass.build(self.test_values, idmap = {})

    def testBuildEmpty(self):
        self.MappedMappingClass.build({})

    def testBuildFromTuples(self):
        self.MappedMappingClass.build(self.test_values)

    def testFileMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedMappingClass.map_file(destfile)
            self.assertMappingOk(mapped)

    def testOrphanMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedMappingClass.map_file(destfile)
        self.assertMappingOk(mapped) # purposedly outside of the tempfile context

    def testBufferMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build(self.test_values, destfile = destfile, idmap = {})
            destfile.seek(0)
            mapped = self.MappedMappingClass.map_buffer(buffer(destfile.read()))
        self.assertMappingOk(mapped)

    def testEmptyMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build({}, destfile = destfile, idmap = {})
            mapped = self.MappedMappingClass.map_file(destfile)
            self.assertMappingOk(mapped, test_values = {})

class MappedMultiMappingTest(MappedMappingTest):
    # Reuse test values from MappedArrayTest to simplify test code
    TEST_VALUES = MappedMappingTest.TEST_VALUES * 2
    TEST_KEYS = MappedMappingTest.TEST_KEYS * 2
    IdMapperClass = mapped_struct.NumericIdMultiMapper

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
        class MappedArrayClass(mapped_struct.MappedArrayProxyBase):
            schema = self.schema
        class MappedMappingClass(mapped_struct.MappedMultiMappingProxyBase):
            ValueArray = MappedArrayClass
            IdMapper = self.IdMapperClass
        self.MappedMappingClass = MappedMappingClass
        self.test_values = [
            (k, self.Struct(**kw))
            for k,kw in zip(self.TEST_KEYS, self.TEST_VALUES)
        ]

    def assertMultivalueContains(self, reference, vals, k):
        for val in vals:
            if self.checkStructEquals(reference, val):
                break
        else:
            self.fail("Cannot find struct in multimap for key %r: expected %r in %r",
                k,
                dict(fset=reference.fset, t=reference.t, l=reference.l),
                [ dict(fset=v.fset, t=v.t, l=v.l) for v in vals ] )

    def assertMappingOk(self, mapping, test_values = None):
        if test_values is None:
            test_values = self.test_values

        # test basic attributes
        self.assertEqual(len(test_values), len(mapping))

        # test key iteration and enumeration
        test_keys = map(operator.itemgetter(0), test_values)
        self.assertEqual(set(test_keys), set(mapping.keys()))
        self.assertEqual(set(test_keys), set(mapping.iterkeys()))

        # test lookup
        for k,reference in test_values:
            self.assertMultivalueContains(reference, mapping[k], k)
        for k,reference in test_values:
            self.assertMultivalueContains(reference, mapping[k], k)

        # test item iteration and enumeration
        for k,proxy in mapping.iteritems():
            reference = [ val for rk,val in self.test_values if rk == k ]
            self.assertMultivalueContains(proxy, reference, k)
        for k,proxy in mapping.items():
            reference = [ val for rk,val in self.test_values if rk == k ]
            self.assertMultivalueContains(proxy, reference, k)

class MappedMappingInt32Test(MappedMappingTest):
    IdMapperClass = mapped_struct.NumericId32Mapper

class MappedMultiMappingInt32Test(MappedMultiMappingTest):
    IdMapperClass = mapped_struct.NumericId32MultiMapper

class MappedStringMappingTest(MappedMappingTest):
    IdMapperClass = mapped_struct.StringIdMapper
    TEST_KEYS = map(str, MappedMappingTest.TEST_KEYS) + [os.urandom(65537)]
    TEST_VALUES = MappedMappingTest.TEST_VALUES + [{
        'fset' : frozenset([(1,2),(3,4),(6,7)]),
        't' : ((3,),(6,7,8),(1,7)),
        'l' : [[1],[2,3],(3,4)],
    }]

class MappedStringMultiMappingTest(MappedMultiMappingTest):
    IdMapperClass = mapped_struct.StringIdMultiMapper
    TEST_KEYS = MappedStringMappingTest.TEST_KEYS * 2
    TEST_VALUES = MappedStringMappingTest.TEST_VALUES * 2

class MappedStringMappingRepeatedValuesTest(MappedStringMappingTest):
    TEST_KEYS = MappedStringMappingTest.TEST_KEYS + map('X2_'.__add__, MappedStringMappingTest.TEST_KEYS)
    TEST_VALUES = MappedStringMappingTest.TEST_VALUES * 2

class MappedString32MappingTest(MappedStringMappingTest):
    IdMapperClass = mapped_struct.StringId32Mapper

class MappedString32MultiMappingTest(MappedStringMultiMappingTest):
    IdMapperClass = mapped_struct.StringId32MultiMapper

class MappedApproxStringMultiMappingTest(MappedStringMultiMappingTest):
    IdMapperClass = mapped_struct.ApproxStringIdMultiMapper

    def testBuildWithDedup(self):
        mapped = self.MappedMappingClass.build(self.test_values, idmap = {}, id_mapper_kwargs = dict(
                discard_duplicates = True
            ))
        self.assertMappingOk(mapped)

    def testBuildWithKeyDedup(self):
        mapped = self.MappedMappingClass.build(self.test_values, idmap = {}, id_mapper_kwargs = dict(
                discard_duplicate_keys = True
            ))
        # Assert works because values of duplicate keys are similar, if they weren't, the test needs to be adapted
        self.assertMappingOk(mapped)

    def assertMappingOk(self, mapping, test_values = None):
        if test_values is None:
            test_values = self.test_values

        # test basic attributes
        self.assertEqual(len(test_values), len(mapping))

        # test lookup
        for k,reference in test_values:
            self.assertMultivalueContains(reference, mapping[k], k)
        for k,reference in test_values:
            self.assertMultivalueContains(reference, mapping[k], k)

        # test item iteration and enumeration
        xxh = mapping.id_mapper.xxh
        encode = mapping.id_mapper.encode
        for k,proxy in mapping.iteritems():
            reference = [ val for rk,val in self.test_values if xxh(encode(rk)).intdigest() == k ]
            self.assertMultivalueContains(proxy, reference, k)
        for k,proxy in mapping.items():
            reference = [ val for rk,val in self.test_values if xxh(encode(rk)).intdigest() == k ]
            self.assertMultivalueContains(proxy, reference, k)

class MappedApproxString32MultiMappingTest(MappedApproxStringMultiMappingTest):
    IdMapperClass = mapped_struct.ApproxStringId32MultiMapper

class MappedStringMappingUnicodeTest(MappedStringMappingTest):
    TEST_KEYS = [ "%s€ ···YEAH···" % (k,) for k in MappedStringMappingTest.TEST_KEYS ]

class MappedString32MappingUnicodeTest(MappedString32MappingTest):
    TEST_KEYS = MappedStringMappingUnicodeTest.TEST_KEYS

class MappedStringMappingBigTest(MappedStringMappingTest):
    TEST_KEYS = [ "%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in xrange(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in xrange(64) ]

class MappedString32MappingBigTest(MappedString32MappingTest):
    TEST_KEYS = [ "%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in xrange(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in xrange(64) ]
