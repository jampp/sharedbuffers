# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
import itertools
import operator
import tempfile
import os
import numpy
import random
import binascii

from sharedbuffers import mapped_struct

try:
    range = xrange
except:
    pass

try:
    unicode
except:
    unicode = str

try:
    import cPickle
except ImportError:
    import pickle as cPickle

try:
    izip = itertools.izip
except AttributeError:
    izip = zip

class SimpleStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

def _make_nattrs(n):
    return dict(
        ('a%d' % i, int)
        for i in range(n)
    )

class Attr7Struct(object):
    __slot_types__ = _make_nattrs(7)
    __slots__ = __slot_types__.keys()

    def __init__(self):
        for k in self.__slot_types__:
            setattr(self, k, 0)

class Attr8Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(8)
    __slots__ = __slot_types__.keys()

class Attr9Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(9)
    __slots__ = __slot_types__.keys()

class Attr15Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(15)
    __slots__ = __slot_types__.keys()

class Attr16Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(16)
    __slots__ = __slot_types__.keys()

class Attr17Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(17)
    __slots__ = __slot_types__.keys()

class Attr31Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(31)
    __slots__ = __slot_types__.keys()

class Attr32Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(32)
    __slots__ = __slot_types__.keys()

class Attr33Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(33)
    __slots__ = __slot_types__.keys()

class Attr63Struct(Attr7Struct):
    __slot_types__ = _make_nattrs(63)
    __slots__ = __slot_types__.keys()

class SizedNumericStruct(object):
    __slot_types__ = {
        'a' : mapped_struct.int32,
        'b' : mapped_struct.float32,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

class PrimitiveStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
        's' : bytes,
        'u' : unicode,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

class ContainerStruct(object):
    __slot_types__ = {
        'fset' : frozenset,
        't' : tuple,
        'l' : list,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

class ObjectStruct(object):
    __slot_types__ = {
        'o' : object,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in kw.items():
            setattr(self, k, v)

class AttributeBitmapTest(unittest.TestCase):
    def _testStruct(self, Struct, delattrs = ()):
        schema = mapped_struct.Schema.from_typed_slots(Struct)
        x = Struct()
        for k in delattrs:
            delattr(x, k)
        dx = schema.unpack(schema.pack(x))
        for k in Struct.__slots__:
            if k in delattrs:
                self.assertFalse(hasattr(dx, k))
            else:
                self.assertEquals(getattr(dx, k, None), getattr(x, k, None))

    def testAttr7(self):
        self._testStruct(Attr7Struct)
    def testAttr8(self):
        self._testStruct(Attr8Struct, delattrs = [ "a%d" % i for i in range(3) ])
    def testAttr9(self):
        self._testStruct(Attr9Struct, delattrs = [ "a%d" % i for i in range(4) ])
    def testAttr15(self):
        self._testStruct(Attr15Struct, delattrs = [ "a%d" % i for i in range(6) ])
    def testAttr16(self):
        self._testStruct(Attr16Struct, delattrs = [ "a%d" % i for i in range(7) ])
    def testAttr17(self):
        self._testStruct(Attr17Struct, delattrs = [ "a%d" % i for i in range(8) ])
    def testAttr31(self):
        self._testStruct(Attr31Struct, delattrs = [ "a%d" % i for i in range(13) ])
    def testAttr32(self):
        self._testStruct(Attr32Struct, delattrs = [ "a%d" % i for i in range(14) ])
    def testAttr33(self):
        self._testStruct(Attr33Struct, delattrs = [ "a%d" % i for i in range(15) ])
    def testAttr63(self):
        self._testStruct(Attr63Struct, delattrs = [ "a%d" % i for i in range(30) ])

class SchemaPicklingTest(AttributeBitmapTest):
    def _testStruct(self, Struct, values = {}, delattrs = ()):
        schema = mapped_struct.Schema.from_typed_slots(Struct)
        x = Struct()

        for k in delattrs:
            delattr(x, k)
        for k,v in values.items():
            setattr(x, k, v)

        px = schema.pack(x)

        old_schema = schema
        schema = cPickle.loads(cPickle.dumps(schema, 2))

        self.assertTrue(old_schema.compatible(schema))
        self.assertTrue(schema.compatible(old_schema))

        dx = schema.unpack(px)
        for k in Struct.__slots__:
            if k in values or k not in delattrs:
                self.assertEquals(getattr(dx, k, None), getattr(x, k, None))
            else:
                self.assertFalse(hasattr(dx, k))

    def testPrimitiveStruct(self):
        self._testStruct(PrimitiveStruct, dict(a=1, b=2.0, s=b'3', u=u'A'))

    def testContainerStruct(self):
        self._testStruct(ContainerStruct, dict(fset=frozenset([3]), t=(1,3), l=[1,2]))

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
            x = self.Struct(**{k:v for k,v in TEST_VALUES.items()})
            dx = self.schema.unpack(self.schema.pack(x))
            for k,v in TEST_VALUES.items():
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testPackPickleUnpack(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in TEST_VALUES.items()})
            pschema = cPickle.loads(cPickle.dumps(self.schema))
            dx = pschema.unpack(self.schema.pack(x))
            for k,v in TEST_VALUES.items():
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testUnpackInto(self):
        dx = self.schema.Proxy()
        for TEST_VALUES in self.TEST_VALUES * 2:
            x = self.Struct(**{k:v for k,v in TEST_VALUES.items()})
            px = self.schema.pack(x)
            dx = self.schema.unpack(px, proxy_into = dx)
            for k,v in TEST_VALUES.items():
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
    doregister = True

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
        if self.doregister:
            # hack - unregister schema
            mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
            mapped_struct.mapped_object.OBJ_PACKERS.pop('}',None)
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

class NestedTypedObjectPackagingTest(NestedObjectPackagingTest):
    SubStruct = ContainerStruct
    subschema = mapped_struct.Schema.from_typed_slots(SubStruct)
    doregister = False

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
        class ContainerObjectStruct(object):
            __slot_types__ = {
                'o' : ContainerStruct,
            }
            __slots__ = __slot_types__.keys()

            def __init__(self, **kw):
                for k,v in kw.items():
                    setattr(self, k, v)
        self.Struct = ContainerObjectStruct

        # hack - unregister schema
        mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop('}',None)

        mapped_struct.mapped_object.register_schema(self.SubStruct, self.subschema, '}')
        super(NestedTypedObjectPackagingTest, self).setUp()

class NestedAutoregisterTypedObjectPackagingTest(NestedTypedObjectPackagingTest):
    def testPackPickleUnpack(self):
        # hack - unregister subschema (can't register twice)
        mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop('}',None)

        for TEST_VALUES in self.TEST_VALUES:
            # re-register subschema
            mapped_struct.mapped_object.register_schema(self.SubStruct, self.subschema, '}')

            x = self.Struct(**{k:v for k,v in TEST_VALUES.items()})
            pschema = cPickle.dumps(self.schema)

            # Unregister schema to force the need for auto-register
            mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
            mapped_struct.mapped_object.OBJ_PACKERS.pop('}',None)

            pschema = cPickle.loads(pschema)

            dx = pschema.unpack(self.schema.pack(x))
            for k,v in TEST_VALUES.items():
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

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
        for reference, proxy in izip(self.test_values, iterator):
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

    def testSchemaReading(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            destfile.seek(0)
            mapped = mapped_struct.MappedArrayProxyBase.map_buffer(buffer(destfile.read()))
            self._checkValues(mapped, mapped)

    def testFastIteration(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedArrayClass.map_file(destfile)
            self._checkValues(mapped, mapped.iter_fast())

    def testFutureCompatible(self):
        with tempfile.NamedTemporaryFile() as destfile:
            class FutureClass(self.MappedArrayClass):
                _CURRENT_VERSION = self.MappedArrayClass._CURRENT_VERSION + 1
            FutureClass.build(self.test_values, destfile = destfile, idmap = {})
            destfile.seek(0)
            mapped = self.MappedArrayClass.map_buffer(buffer(destfile.read()))
            self._checkValues(mapped, mapped)

    def testFutureIncompatible(self):
        with tempfile.NamedTemporaryFile() as destfile:
            class FutureClass(self.MappedArrayClass):
                _CURRENT_VERSION = _CURRENT_MINIMUM_READER_VERSION = self.MappedArrayClass._CURRENT_VERSION + 1
            FutureClass.build(self.test_values, destfile = destfile, idmap = {})
            destfile.seek(0)
            self.assertRaises(ValueError, self.MappedArrayClass.map_buffer, buffer(destfile.read()))

class IdMapperTest(unittest.TestCase):
    IdMapperClass = mapped_struct.NumericIdMapper

    def gen_values(self, n, reversed = False, shuffled = False, gen_dupes = False):
        if reversed:
            keys = range(n-1,-1,-1)
        else:
            keys = range(n)
        if shuffled:
            keys = list(keys)
            r = random.Random(1234827)
            r.shuffle(keys)
        if gen_dupes:
            return itertools.chain(
                izip(keys, range(0, 2*n, 2)),
                itertools.islice(izip(keys, range(0, 2*n, 2)), 10, None),
            )
        else:
            return izip(keys, range(0, 2*n, 2))

    def _testBuild(self, N, tempdir, **gen_kwargs):
        build_kwargs = gen_kwargs.pop('build_kwargs', {})
        rv = self.IdMapperClass.build(self.gen_values(N, **gen_kwargs), tempdir = tempdir, **build_kwargs)
        rvget = rv.get
        for k, v in self.gen_values(N, **gen_kwargs):
            rvv = rvget(k)
            if rvv != v:
                self.assertEquals(rvv, v)

    def testBuildHugeInMem(self):
        self._testBuild(2010530, None)

    def testBuildHugeInMemReversed(self):
        self._testBuild(2010530, None, reversed = True)

    def testBuildHugeInMemShuffled(self):
        self._testBuild(2010530, None, shuffled = True)

    def testBuildHugeInMemDiscardDuplicates(self):
        self._testBuild(2010530, None, build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

    def testBuildHugeOnDisk(self):
        self._testBuild(10107530, tempfile.gettempdir())

    def testBuildHugeOnDiskReversed(self):
        self._testBuild(10107530, tempfile.gettempdir(), reversed=True)

    def testBuildHugeOnDiskShuffled(self):
        self._testBuild(10107530, tempfile.gettempdir(), shuffled=True)

    def testBuildHugeOnDiskDiscardDuplicates(self):
        self._testBuild(10107530, tempfile.gettempdir(), build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

class Id32MapperTest(IdMapperTest):
    IdMapperClass = mapped_struct.NumericId32Mapper

class ApproxStringIdMultiMapperTest(IdMapperTest):
    IdMapperClass = mapped_struct.ApproxStringIdMultiMapper

    def gen_values(self, *p, **kw):
        str_ = str
        for k, v in super(ApproxStringIdMultiMapperTest, self).gen_values(*p, **kw):
            yield str_(k), v

    def _testBuild(self, N, tempdir, **gen_kwargs):
        build_kwargs = gen_kwargs.pop('build_kwargs', {})
        rv = self.IdMapperClass.build(self.gen_values(N, **gen_kwargs), tempdir = tempdir, **build_kwargs)
        str_ = str
        rvget = rv.get
        for k, v in self.gen_values(N, **gen_kwargs):
            elem = rvget(str_(k))
            if elem is None:
                self.assertIsNotNone(elem)
            if v not in elem:
                self.assertIn(v, elem)

    # Too much memory
    testBuildHugeInMemShuffled = None
    testBuildHugeOnDiskShuffled = None

class ApproxStringId32MultiMapperTest(ApproxStringIdMultiMapperTest):
    IdMapperClass = mapped_struct.ApproxStringId32MultiMapper

class MappedMappingTest(unittest.TestCase):
    # Reuse test values from MappedArrayTest to simplify test code
    Struct = MappedArrayTest.Struct
    TEST_VALUES = MappedArrayTest.TEST_VALUES
    TEST_KEYS = [ x*7 for x in range(len(TEST_VALUES)) ]
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
        for k,reference in test_values.items():
            self.assertStructEquals(reference, mapping.get(k))
        for k,reference in test_values.items():
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
            self.fail("Cannot find struct in multimap for key %r: expected %r in %r" % (
                k,
                dict(fset=reference.fset, t=reference.t, l=reference.l),
                [ dict(fset=v.fset, t=v.t, l=v.l) for v in vals ]) )

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
            self.assertIn(k, mapping)
            self.assertMultivalueContains(reference, mapping[k], k)
            self.assertMultivalueContains(reference, mapping.get(k), k)
            self.assertMultivalueContains(reference, list(mapping.get_iter(k)), k)
        for k,reference in test_values:
            self.assertIn(k, mapping)
            self.assertMultivalueContains(reference, mapping[k], k)
            self.assertMultivalueContains(reference, mapping.get(k), k)
            self.assertMultivalueContains(reference, list(mapping.get_iter(k)), k)

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
    TEST_KEYS = list(map(str, MappedMappingTest.TEST_KEYS)) + [str(binascii.hexlify(os.urandom(65537)))]
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
    TEST_KEYS = MappedStringMappingTest.TEST_KEYS + list(map('X2_'.__add__, MappedStringMappingTest.TEST_KEYS))
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
            self.assertIn(k, mapping)
            self.assertMultivalueContains(reference, mapping[k], k)
            self.assertMultivalueContains(reference, mapping.get(k), k)
            self.assertMultivalueContains(reference, list(mapping.get_iter(k)), k)
        for k,reference in test_values:
            self.assertIn(k, mapping)
            self.assertMultivalueContains(reference, mapping[k], k)
            self.assertMultivalueContains(reference, mapping.get(k), k)
            self.assertMultivalueContains(reference, list(mapping.get_iter(k)), k)

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
    TEST_KEYS = [ "%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in range(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in range(64) ]

class MappedString32MappingBigTest(MappedString32MappingTest):
    TEST_KEYS = [ "%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in range(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in range(64) ]

class BsearchTest(unittest.TestCase):
    if mapped_struct._cythonized:
        SUPPORTED_DTYPES = [ numpy.uint32, numpy.int32, numpy.uint64, numpy.int64,
            numpy.double, numpy.single, numpy.float64, numpy.float32 ]

        UNSUPPORTED_DTYPES = [ numpy.uint16, numpy.int16, numpy.uint8, numpy.int8 ]
    else:
        SUPPORTED_DTYPES = [ numpy.uint32, numpy.int32, numpy.uint64, numpy.int64,
            numpy.double, numpy.single, numpy.float64, numpy.float32,
            numpy.uint16, numpy.int16, numpy.uint8, numpy.int8 ]

        UNSUPPORTED_DTYPES = []

    for dtype in SUPPORTED_DTYPES:
        def testBsearch(self, dtype=dtype):
            testarray = list(range(1,101))
            random.shuffle(testarray)
            a = numpy.array(testarray[:50], dtype)
            b = numpy.array([0] + testarray[50:] + list(range(101,103)), dtype)
            a = numpy.sort(a)
            self.assertEqual(mapped_struct.bsearch(a, 0), 0)
            self.assertEqual(mapped_struct.bsearch(a, 101), len(a))
            self.assertEqual(mapped_struct.bsearch(a, 102), len(a))
            for x in a:
                ix = mapped_struct.bsearch(a, x)
                self.assertLess(ix, len(a))
                self.assertEqual(a[ix], x)
                self.assertTrue(mapped_struct.sorted_contains(a, x))
            for x in b:
                ix = mapped_struct.bsearch(a, x)
                self.assertTrue(ix >= len(a) or a[ix] != x)
                self.assertFalse(mapped_struct.sorted_contains(a, x))
        testBsearch.__name__ += dtype.__name__.title()
        locals()[testBsearch.__name__] = testBsearch
        del testBsearch
        del dtype

    for dtype in UNSUPPORTED_DTYPES:
        def testBsearchUnsupported(self, dtype=dtype):
            a = numpy.arange(50, dtype=dtype)
            for x in a:
                self.assertRaises(NotImplementedError, mapped_struct.bsearch, a, x)
        testBsearchUnsupported.__name__ += dtype.__name__.title()
        locals()[testBsearchUnsupported.__name__] = testBsearchUnsupported
        del testBsearchUnsupported
        del dtype

    def testBsearchDuplicates(self):
        a = numpy.array([1,1,2,2,3,4,4,5,6,6])
        self.assertEqual(0, mapped_struct.bsearch(a, 1))
        self.assertEqual(2, mapped_struct.bsearch(a, 2))
        self.assertEqual(5, mapped_struct.bsearch(a, 4))
        self.assertEqual(8, mapped_struct.bsearch(a, 6))

    def testBsearchEmpty(self):
        a = numpy.array([], dtype=numpy.uint32)
        self.assertEqual(0, mapped_struct.bsearch(a, 1))
        self.assertEqual(0, mapped_struct.bsearch(a, 2))
        self.assertEqual(0, mapped_struct.bsearch(a, 4))
        self.assertEqual(0, mapped_struct.bsearch(a, 6))

class MergeTest(unittest.TestCase):
    if mapped_struct._cythonized:
        SUPPORTED_DTYPES = [ numpy.uint32, numpy.int32, numpy.uint64, numpy.int64,
            numpy.double, numpy.single, numpy.float64, numpy.float32 ]

        UNSUPPORTED_DTYPES = [ numpy.uint16, numpy.int16, numpy.uint8, numpy.int8 ]
    else:
        SUPPORTED_DTYPES = [ numpy.uint32, numpy.int32, numpy.uint64, numpy.int64,
            numpy.double, numpy.single, numpy.float64, numpy.float32,
            numpy.uint16, numpy.int16, numpy.uint8, numpy.int8 ]

        UNSUPPORTED_DTYPES = []

    for dtype in SUPPORTED_DTYPES:
        def testMerge(self, dtype=dtype):
            testarray1 = list(range(1,101))
            testarray2 = list(range(5,106))
            a = numpy.empty((100,2), dtype=dtype)
            b = numpy.empty((100,2), dtype=dtype)
            merged = numpy.empty((200,2), dtype=dtype)
            incompatible1 = numpy.empty((200,3), dtype=dtype)
            incompatible2 = numpy.empty(200, dtype=dtype)
            a[:,0] = numpy.arange(1,101)
            a[:,1] = numpy.arange(2,102)
            b[:,0] = numpy.arange(5,105)
            b[:,1] = numpy.arange(6,106)
            ref = numpy.concatenate([a,b])
            ref = ref[numpy.argsort(ref[:,0])]
            self.assertEqual(mapped_struct.index_merge(a, b, merged), 200)
            self.assertTrue((merged == ref).all())
            self.assertRaises(ValueError, mapped_struct.index_merge, a, b, incompatible1)
            self.assertRaises(ValueError, mapped_struct.index_merge, a, incompatible1, merged)
            self.assertRaises(ValueError, mapped_struct.index_merge, a, b, incompatible2)
            self.assertRaises(ValueError, mapped_struct.index_merge, a, incompatible2, merged)
        testMerge.__name__ += dtype.__name__.title()
        locals()[testMerge.__name__] = testMerge
        del testMerge
        del dtype

    for dtype in UNSUPPORTED_DTYPES:
        def testMergeUnsupported(self, dtype=dtype):
            a = numpy.empty((50,2), dtype=dtype)
            dest = numpy.empty((100,2), dtype=dtype)
            self.assertRaises(NotImplementedError, mapped_struct.index_merge, a, a, dest)
        testMergeUnsupported.__name__ += dtype.__name__.title()
        locals()[testMergeUnsupported.__name__] = testMergeUnsupported
        del testMergeUnsupported
        del dtype

    def testRejectInPlace(self):
        a = numpy.empty(dtype=numpy.uint32, shape = [0,2])
        self.assertRaises(NotImplementedError, mapped_struct.index_merge, a, a, a)

    def testRejectOverlapping(self):
        x = numpy.empty(dtype=numpy.uint32, shape = [4,2])

        a = x[:1]
        b = x[:1]
        d = x
        self.assertRaises(NotImplementedError, mapped_struct.index_merge, a, b, d)

        a = x[:1]
        b = x[2:3]
        d = x[1:]
        self.assertRaises(NotImplementedError, mapped_struct.index_merge, a, b, d)

    def testMergeEmpty(self):
        a = numpy.empty(shape=[0,2], dtype=numpy.uint32)
        d = numpy.empty(shape=[1,2], dtype=numpy.uint32)
        self.assertEqual(0, mapped_struct.index_merge(a, a, d))

class IdMapperMergeTest(unittest.TestCase):
    NUMERIC_TEST_1 = [ (1,2), (3,4), (5,6) ]
    NUMERIC_TEST_2 = [ (2,3), (4,5), (6,7) ]
    NUMERIC_TEST_3 = [ (10,11), (12,13), (14,15) ]
    NUMERIC_TEST_4 = [ (10,12), (12,14), (14,16) ]

    STRING_TEST_1 = [ ('1',2), ('3',4), ('5',6) ]
    STRING_TEST_2 = [ ('2',3), ('4',5), ('6',7) ]
    STRING_TEST_3 = [ ('10',11), ('12',13), ('14',15) ]
    STRING_TEST_4 = [ ('10',12), ('12',14), ('14',16) ]

    def _testMerge(self, cls, part_data):
        parts = [ cls.build(seq) for seq in part_data ]
        merged = cls.merge(parts)
        all_items = set()
        for seq in part_data:
            all_items |= set(seq)
            for k,v in seq:
                self.assertEqual(merged[k], v)
        self.assertEqual(set(merged.items()), all_items)

    def _testMergeMulti(self, cls, part_data):
        parts = [ cls.build(seq) for seq in part_data ]
        merged = cls.merge(parts)
        all_items = set()
        for seq in part_data:
            all_items |= set(seq)
            for k,v in seq:
                self.assertIn(v, merged[k])

    def testMergeNumericIdMapper(self):
        self._testMerge(mapped_struct.NumericIdMapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2])

    def testMergeNumericIdMapper3(self):
        self._testMerge(mapped_struct.NumericIdMapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2, self.NUMERIC_TEST_3])

    def testMergeNumericId32Mapper(self):
        self._testMerge(mapped_struct.NumericId32Mapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2])

    def testMergeNumericId32Mapper3(self):
        self._testMerge(mapped_struct.NumericId32Mapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2, self.NUMERIC_TEST_3])

    def testMergeNumericIdMultiMapper(self):
        self._testMergeMulti(mapped_struct.NumericIdMultiMapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2, self.NUMERIC_TEST_3, self.NUMERIC_TEST_4])

    def testMergeNumericId32MultiMapper(self):
        self._testMergeMulti(mapped_struct.NumericId32MultiMapper,
            [self.NUMERIC_TEST_1, self.NUMERIC_TEST_2, self.NUMERIC_TEST_3, self.NUMERIC_TEST_4])

    def testMergeApproxStringIdMultiMapper(self):
        self._testMergeMulti(mapped_struct.ApproxStringIdMultiMapper,
            [self.STRING_TEST_1, self.STRING_TEST_2])

    def testMergeApproxStringIdMultiMapper4(self):
        self._testMergeMulti(mapped_struct.ApproxStringIdMultiMapper,
            [self.STRING_TEST_1, self.STRING_TEST_2, self.STRING_TEST_3, self.STRING_TEST_4])

    def testMergeApproxStringId32MultiMapper(self):
        self._testMergeMulti(mapped_struct.ApproxStringId32MultiMapper,
            [self.STRING_TEST_1, self.STRING_TEST_2])

    def testMergeApproxStringId32MultiMapper4(self):
        self._testMergeMulti(mapped_struct.ApproxStringId32MultiMapper,
            [self.STRING_TEST_1, self.STRING_TEST_2, self.STRING_TEST_3, self.STRING_TEST_4])

class FrozensetPackingTest(unittest.TestCase):
    def testUnpackOffBounds(self):
        b = buffer("")
        self.assertRaises(IndexError, mapped_struct.mapped_frozenset.unpack_from, b, 5)

    def testUnpackBeyondEnd(self):
        b = buffer("m")
        self.assertRaises(IndexError, mapped_struct.mapped_frozenset.unpack_from, b, 0)

    def testSingletons(self):
        a = bytearray(16)
        fs = frozenset()
        mapped_struct.mapped_frozenset.pack_into(fs, a, 0)
        self.assertIs(mapped_struct.mapped_frozenset.unpack_from(a, 0), fs)
