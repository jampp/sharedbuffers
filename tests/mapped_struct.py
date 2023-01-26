# -*- coding: utf-8 -*-
from __future__ import absolute_import

import time
import unittest
import itertools
import operator
import tempfile
import os
import numpy
import random
import zipfile
import dateutil.tz
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from decimal import Decimal
import numpy as np
import six

if six.PY3:
    long = int
    buffer = memoryview

try:
    from cdecimal import Decimal as cDecimal
except:
    cDecimal = Decimal

from sharedbuffers import mapped_struct
if six.PY3:
    from sharedbuffers import cmp
import functools
from six import iteritems, itervalues, iterkeys
from six.moves import xrange, zip, zip_longest

SKIP_HUGE = os.environ.get('SKIP_HUGE','')

from six.moves import cPickle

SKIP_HUGE = os.environ.get('SKIP_HUGE', None)


@contextmanager
def set_tz(tz):
    og_tz = os.environ.get("TZ")

    try:
        os.environ["TZ"] = tz
        time.tzset()
        yield
    finally:
        if og_tz is not None:
            os.environ["TZ"] = og_tz
            time.tzset()


class SimpleStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
    }
    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in iteritems(kw):
            setattr(self, k, v)

def _make_nattrs(n):
    return dict(
        ('a%d' % i, int)
        for i in xrange(n)
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

class TestStruct(object):

    __slot_types__ = {}

    __slots__ = __slot_types__.keys()

    def __init__(self, **kw):
        for k,v in iteritems(kw):
            setattr(self, k, v)

class SizedNumericStruct(TestStruct):
    __slot_types__ = {
        'a' : mapped_struct.int32,
        'b' : mapped_struct.float32,
    }

class PrimitiveStruct(TestStruct):
    __slot_types__ = {
        'a' : int,
        'b' : float,
        's' : bytes,
        'u' : six.text_type,
    }

class DatetimeStruct(TestStruct):
    __slot_types__ = {
        'd' : datetime,
        'D' : date,
    }

class DecimalStruct(TestStruct):
    __slot_types__ = {
        'd' : Decimal,
        'D' : cDecimal,
    }

class BufferStruct(TestStruct):
    __slot_types__ = {
        'b' : buffer,
    }

class ContainerStruct(TestStruct):
    __slot_types__ = {
        'fset' : frozenset,
        't' : tuple,
        'l' : list,
        'pt': mapped_struct.proxied_tuple,
        'pl': mapped_struct.proxied_list,
    }

class DictStruct(TestStruct):
    __slot_types__ = {
        'd' : dict,
        'D' : mapped_struct.proxied_dict
    }

class NDArrayStruct(TestStruct):
    __slot_types__ = {
        'a' : np.ndarray,
    }

class ObjectStruct(TestStruct):
    __slot_types__ = {
        'b' : bytes,
        'q' : bytes,
        'o' : object,
    }

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

    def _testStruct(self, Struct, values = {}, delattrs = (), cmp_func = None):
        schema = mapped_struct.Schema.from_typed_slots(Struct)
        x = Struct()

        for k in delattrs:
            delattr(x, k)
        for k,v in iteritems(values):
            setattr(x, k, v)

        px = schema.pack(x)

        old_schema = schema
        schema = cPickle.loads(cPickle.dumps(schema, 2))

        self.assertTrue(old_schema.compatible(schema))
        self.assertTrue(schema.compatible(old_schema))

        dx = schema.unpack(px)
        for k in Struct.__slots__:
            if k in values or k not in delattrs:
                if cmp_func:
                    self.assertTrue(cmp_func(getattr(dx, k, None), getattr(x, k, None)))
                else:
                    self.assertTrue(getattr(dx, k, None) == getattr(x, k, None))
            else:
                self.assertFalse(hasattr(dx, k))

    def testPrimitiveStruct(self):
        self._testStruct(PrimitiveStruct, dict(a=1, b=2.0, s=b'3', u=u'A'))

    def testDictStruct(self):
        self._testStruct(DictStruct, dict(d={"a":1, "b":2}, D={"a": 1, "b": 2}))

    def testDatetimeStruct(self):
        self._testStruct(DatetimeStruct, dict(d=datetime.now(), D=date.today()))

    def testBufferStruct(self):
        self._testStruct(BufferStruct, dict(b=buffer(bytearray(xrange(100)))))

    def testNDArrayStruct(self):
        self._testStruct(DecimalStruct, dict(a=np.array([-1.0, 2.5, 3])), cmp_func=np.array_equal)

    def testDecimalStruct(self):
        cmp_func = lambda a, b: str(a) == str(b)
        self._testStruct(DecimalStruct, dict(d=Decimal(1.23), D=cDecimal(1.245)), cmp_func=cmp_func)

    def testCastIntToDecimalStruct(self):
        cmp_func = lambda a, b: int(a) == int(b)
        self._testStruct(DecimalStruct, dict(d=1, D=2), cmp_func=cmp_func)

    def testCastFloatToDecimalStruct(self):
        cmp_func = lambda a, b: float(a) == float(b)
        self._testStruct(DecimalStruct, dict(d=1.23, D=1.245), cmp_func=cmp_func)

    def testContainerStruct(self):
        self._testStruct(ContainerStruct, dict(fset=frozenset([3]), t=(1,3), l=[1,2], pt=(1.0,2.0), pl=[1.0,2.0]))

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
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            dx = self.schema.unpack(self.schema.pack(x))
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testPackUnpackIdmap(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            pack_idmap = {}
            unpack_idmap = {}
            dx = self.schema.unpack(self.schema.pack(x, pack_idmap), unpack_idmap)
            for k,v in iteritems(TEST_VALUES):
                unpack_idmap.clear()
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                unpack_idmap.clear()
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testPackMultiple(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})

            buf = bytearray(16<<20)
            offsets = []
            endp = 0
            for i in xrange(10):
                offsets.append(endp)
                endp = self.schema.pack_into(x, buf, endp)
            for offs in offsets:
                dx = self.schema.unpack_from(buf, offs)
                for k,v in iteritems(TEST_VALUES):
                    self.assertTrue(hasattr(dx, k))
                    self.assertEqual(getattr(dx, k), v)
                for k in self.Struct.__slots__:
                    if k not in TEST_VALUES:
                        self.assertFalse(hasattr(dx, k))

    def testPackPickleUnpack(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            pschema = cPickle.loads(cPickle.dumps(self.schema))
            dx = pschema.unpack(self.schema.pack(x))
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testUnpackInto(self):
        dx = self.schema.Proxy()
        for TEST_VALUES in self.TEST_VALUES * 2:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            px = self.schema.pack(x)
            dx = self.schema.unpack(px, proxy_into = dx)
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

    def testPackRepack(self):
        for TEST_VALUES in self.TEST_VALUES:
            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            dx = self.schema.unpack(self.schema.pack(x))
            dx = self.schema.unpack(self.schema.pack(dx))
            for k,v in iteritems(TEST_VALUES):
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
        'pt' : (3,6,7),
        'pl' : [1,2,3],
    }]

class ShortIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1000,3000,7000]),
        't' : (3000,6000,7000),
        'l' : [1000,2000,3000],
        'pt' : (3000,6000,7000),
        'pl' : [1000,2000,3000],
    }]

class LongIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([0x12341234,0x23452345,0x43534353]),
        't' : (0x12241234,0x23352345,0x43334353),
        'l' : [0x13341234,0x24452345,0x45534353],
        'pt' : (0x12241234,0x23352345,0x43334353),
        'pl' : [0x13341234,0x24452345,0x45534353],
    }]

class LongLongIntContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([0xf012341234,0xf023452345,0xf043534353]),
        't' : (0xf012241234,0xf023352345,0xf043334353),
        'l' : [0xf013341234,0xf024452345,0xf045534353],
        'pt' : (0xf012241234,0xf023352345,0xf043334353),
        'pl' : [0xf013341234,0xf024452345,0xf045534353],
    }]

class FloatContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1.0,3.0,7.0]),
        't' : (3.0,6.0,7.0),
        'l' : [1.0,2.0,3.0],
        'pt' : (3.0,6.0,7.0),
        'pl' : [1.0,2.0,3.0],
    }]

class BytesContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([b'1.0',b'3.0',b'7.0']),
        't' : (b'3.0',b'6.0',b'7.0'),
        'l' : [b'1.0',b'2.0',b'3.0'],
        'pt' : (b'3.0',b'6.0',b'7.0'),
        'pl' : [b'1.0',b'2.0',b'3.0'],
    }]

class NestedContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([(1,2),(3,4),(6,7)]),
        't' : ((3,),(6,7,8),(1,7)),
        'l' : [[1],[2,3],(3,4)],
        'pt' : ((3,),(6,7,8),(1,7)),
        'pl' : [[1],[2,3],(3,4)],
    }]

class NumpyCastingContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : numpy.array([1,3,7], dt),
        't' : numpy.array([3,6,7], dt),
        'l' : numpy.array([1,2,3], dt),
        'pt' : numpy.array([3,6,7], dt),
        'pl' : numpy.array([1,2,3], dt),
    } for dt in (
        numpy.int8,
        numpy.uint8,
        numpy.int16,
        numpy.uint8,
        numpy.int32,
        numpy.uint32,
        numpy.int64,
        numpy.uint64,
        numpy.float,
        numpy.double,
    )]

    def assertEqual(self, a, b, *p, **kw):
        if type(a) is not type(b):
            typemap = {
                mapped_struct.proxied_list : list,
                mapped_struct.proxied_tuple : tuple,
            }
            if isinstance(a, numpy.ndarray):
                btype = type(b)
                btype = typemap.get(btype, btype)
                a = btype(a)
            elif isinstance(b, numpy.ndarray):
                atype = type(a)
                atype = typemap.get(atype, atype)
                b = atype(b)
        return super(NumpyCastingContainerPackingTest, self).assertEqual(a, b, *p, **kw)

class DictContainerPackingTest(SimplePackingTest):
    Struct = ContainerStruct
    TEST_VALUES = [{
        'fset' : frozenset([1,3,7]),
        't' : ({'a':frozenset([1,3,4])},{'b':(1,3,4)},{'c':[1,3,4]}),
        'l' : [{'a':frozenset([1,3,4])},{'b':(1,3,4)},{'c':[1,3,4]}],
        'pt' : ({'a':frozenset([1,3,4])},{'b':(1,3,4)},{'c':[1,3,4]}),
        'pl' : [{'a':frozenset([1,3,4])},{'b':(1,3,4)},{'c':[1,3,4]}],
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
        { 'o' : b"blabla" },
        { 'o' : u"bláblá€" },
        { 'o' : datetime.now() },
        { 'o' : date.today() },
        { 'o' : buffer(bytearray(xrange(100)))},
        { 'b' : b"blabla", 'o' : b"blabla" },
        { 'q' : b"blabla", 'o' : b"blabla" },
        { 'b' : b"blabla", 'o' : b"blabla", 'q' : b"blabla" },
    ]

class ObjectNDArrayPackagingTest(SimplePackingTest):
    Struct = ObjectStruct

    def assertEqual(self, v1, v2):
        self.assertTrue(np.array_equal(v1, v2))

    TEST_VALUES = [
        { 'o' : np.array([-1.0, 2.5, 3]) },
    ]

class ObjectDecimalPackagingTest(SimplePackingTest):
    Struct = ObjectStruct

    def assertEqual(self, v1, v2):
        self.assertTrue(str(v1) == str(v2))

    TEST_VALUES = [
        { 'o' : Decimal(123.456) },
        { 'o' : cDecimal(123.456) },
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
                'pt' : (3000,6000,7000),
                'pl' : [1000,2000,3000],
            }),
        },
    ]

    def setUp(self):
        if self.doregister:
            # hack - unregister schema
            mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
            mapped_struct.mapped_object.OBJ_PACKERS.pop(b'}',None)
            mapped_struct.mapped_object.register_schema(self.SubStruct, self.subschema, b'{')
        super(NestedObjectPackagingTest, self).setUp()

    def assertEqual(self, value, expected, *p, **kw):
        if isinstance(expected, self.SubStruct):
            for k in self.SubStruct.__slots__:
                self.assertEqual(hasattr(value, k), hasattr(expected, k), *p, **kw)
                if hasattr(expected, k):
                    self.assertEqual(getattr(value, k), getattr(expected, k), *p, **kw)
        else:
            return super(NestedObjectPackagingTest, self).assertEqual(value, expected, *p, **kw)

    # Not supported without a registered schema
    testPackRepack = None

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
                'pt' : (3000,6000,7000),
                'pl' : [1000,2000,3000],
            }),
        },
    ]

    # Reinstate testPackRepack
    testPackRepack = SimplePackingTest.testPackRepack

    def setUp(self):
        class ContainerObjectStruct(object):
            __slot_types__ = {
                'o' : ContainerStruct,
            }
            __slots__ = __slot_types__.keys()

            def __init__(self, **kw):
                for k,v in iteritems(kw):
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

            x = self.Struct(**{k:v for k,v in iteritems(TEST_VALUES)})
            pschema = cPickle.dumps(self.schema)

            # Unregister schema to force the need for auto-register
            mapped_struct.mapped_object.TYPE_CODES.pop(self.SubStruct,None)
            mapped_struct.mapped_object.OBJ_PACKERS.pop('}',None)

            pschema = cPickle.loads(pschema)

            dx = pschema.unpack(self.schema.pack(x))
            for k,v in iteritems(TEST_VALUES):
                self.assertTrue(hasattr(dx, k))
                self.assertEqual(getattr(dx, k), v)
            for k in self.Struct.__slots__:
                if k not in TEST_VALUES:
                    self.assertFalse(hasattr(dx, k))

class RegistryConflictTest(unittest.TestCase):
    Struct = Attr8Struct
    schema = mapped_struct.Schema.from_typed_slots(Struct)

    def setUp(self):
        # hack - make sure registry doesn't get screwed
        self.addCleanup(
            mapped_struct.mapped_object.OBJ_PACKERS.__setitem__,
            b't', mapped_struct.mapped_object.OBJ_PACKERS[b't'])
        self.addCleanup(
            mapped_struct.mapped_object.PACKERS.__setitem__,
            b'b', mapped_struct.mapped_object.PACKERS[b'b'])

    def testRegistryConflictSchema(self):
        with self.assertRaises(ValueError):
            mapped_struct.mapped_object.register_schema(self.Struct, self.schema, b't')

    def testRegistryConflictBuiltin(self):
        with self.assertRaises(ValueError):
            mapped_struct.mapped_object.register_schema(self.Struct, self.schema, b'b')

class MappedArrayTest(unittest.TestCase):
    Struct = ContainerStruct
    TEST_VALUES = [
        {
            'fset' : frozenset(['1.0','3.0','7.0']),
            't' : ('3.0','6.0','7.0'),
            'l' : ['1.0','2.0','3.0'],
            'pt' : ('3.0','6.0','7.0'),
            'pl' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset([1,3,7]),
            't' : (3,6,7),
            'l' : [1,2,3],
            'pt' : (3,6,7),
            'pl' : [1,2,3],
        }, {
            'fset' : frozenset([1000,3000,7000]),
            't' : (3000,6000,7000),
            'l' : [1000,2000,3000],
            'pt' : (3000,6000,7000),
            'l' : [1000,2000,3000],
        }, {
            'fset' : frozenset([0x12341234,0x23452345,0x43534353]),
            't' : (0x12241234,0x23352345,0x43334353),
            'l' : [0x13341234,0x24452345,0x45534353],
            'pt' : (0x12241234,0x23352345,0x43334353),
            'pl' : [0x13341234,0x24452345,0x45534353],
        }, {
            'fset' : frozenset([0xf012341234,0xf023452345,0xf043534353]),
            't' : (0xf012241234,0xf023352345,0xf043334353),
            'l' : [0xf013341234,0xf024452345,0xf045534353],
            'pt' : (0xf012241234,0xf023352345,0xf043334353),
            'pl' : [0xf013341234,0xf024452345,0xf045534353],
        }, {
            'fset' : frozenset([1.0,3.0,7.0]),
            't' : (3.0,6.0,7.0),
            'l' : [1.0,2.0,3.0],
            'pt' : (3.0,6.0,7.0),
            'pl' : [1.0,2.0,3.0],
        }, {
            'fset' : frozenset(['1.0','3.0','7.0']),
            't' : ('3.0','6.0','7.0'),
            'l' : ['1.0','2.0','3.0'],
            'pt' : ('3.0','6.0','7.0'),
            'pl' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset(['1.0',os.urandom(128000),'7.0']),
            't' : ('3.0',os.urandom(256000),'7.0'),
            'l' : ['1.0','2.0','3.0'],
            'pt' : ('3.0',os.urandom(256000),'7.0'),
            'pl' : ['1.0','2.0','3.0'],
        }, {
            'fset' : frozenset([(1,2),(3,4),(6,7)]),
            't' : ((3,),(6,7,8),(1,7)),
            'l' : [[1],[2,3],(3,4)],
            'pt' : ((3,),(6,7,8),(1,7)),
            'pl' : [[1],[2,3],(3,4)],
        }
    ]

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
        class MappedArrayClass(mapped_struct.MappedArrayProxyBase):
            schema = self.schema
        self.MappedArrayClass = MappedArrayClass
        self.test_values = [ self.Struct(**kw) for kw in self.TEST_VALUES ]

    def _checkValues(self, mapping, iterator, slice=None):
        self.assertEqual(len(self.test_values), len(mapping))
        test_values = self.test_values
        if slice is not None:
            test_values = test_values[slice]
        for reference, proxy in zip_longest(test_values, iterator):
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

    def testZipMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            with tempfile.NamedTemporaryFile() as tempzip:
                zf = zipfile.ZipFile(tempzip, 'w')
                zf.write(destfile.name, 'bundle', zipfile.ZIP_STORED)
                zf.writestr('otherdata', 'blablabla')
                zf.close()

                tempzip.seek(0)
                zf = zipfile.ZipFile(tempzip, 'r')
                mapped = self.MappedArrayClass.map_file(zf.open('bundle'))
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

    def testMaskedIteration(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedArrayClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedArrayClass.map_file(destfile)
            mask = numpy.zeros(len(mapped), numpy.uint8)
            mask[::2] = True
            self._checkValues(mapped, mapped.iter_fast(mask=mask), slice(None, None, 2))

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
            keys = xrange(n-1,-1,-1)
        else:
            keys = xrange(n)
        if shuffled:
            keys = list(keys)
            r = random.Random(1234827)
            r.shuffle(keys)
        if gen_dupes:
            return itertools.chain(
                zip(keys, xrange(0, 2*n, 2)),
                itertools.islice(zip(keys, xrange(0, 2*n, 2)), 10, None),
            )
        else:
            return zip(keys, xrange(0, 2*n, 2))

    def _testBuild(self, N, tempdir, **gen_kwargs):
        build_kwargs = gen_kwargs.pop('build_kwargs', {})
        rv = self.IdMapperClass.build(self.gen_values(N, **gen_kwargs), tempdir = tempdir, **build_kwargs)
        rvget = rv.get
        for k, v in self.gen_values(N, **gen_kwargs):
            rvv = rvget(k)
            if rvv != v:
                self.assertEquals(rvv, v)

    def testBuildEmpty(self):
        rv = self.IdMapperClass.build([], tempdir = None)
        rvget = rv.get
        for k, v in self.gen_values(10):
            self.assertNotIn(k, rv)
            rvv = rvget(k)
            self.assertIsNone(rvv)

        self.assertEquals(list(rv), [])
        self.assertEquals(list(rv.keys()), [])
        self.assertEquals(list(rv.values()), [])
        self.assertEquals(list(rv.items()), [])

    def testBuildInMem(self):
        self._testBuild(2010, None)

    def testBuildInMemReversed(self):
        self._testBuild(2010, None, reversed = True)

    def testBuildInMemShuffled(self):
        self._testBuild(2010, None, shuffled = True)

    def testBuildInMemDiscardDuplicates(self):
        self._testBuild(2010, None, build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

    def testBuildOnDisk(self):
        self._testBuild(10107, tempfile.gettempdir())

    def testBuildOnDiskReversed(self):
        self._testBuild(10107, tempfile.gettempdir(), reversed=True)

    def testBuildOnDiskShuffled(self):
        self._testBuild(10107, tempfile.gettempdir(), shuffled=True)

    def testBuildOnDiskDiscardDuplicates(self):
        self._testBuild(10107, tempfile.gettempdir(), build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeInMem(self):
        self._testBuild(2010530, None)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeInMemReversed(self):
        self._testBuild(2010530, None, reversed = True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeInMemShuffled(self):
        self._testBuild(2010530, None, shuffled = True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeInMemDiscardDuplicates(self):
        self._testBuild(2010530, None, build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeOnDisk(self):
        self._testBuild(10107530, tempfile.gettempdir())

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeOnDiskReversed(self):
        self._testBuild(10107530, tempfile.gettempdir(), reversed=True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeOnDiskShuffled(self):
        self._testBuild(10107530, tempfile.gettempdir(), shuffled=True)

    @unittest.skipIf(SKIP_HUGE, 'SKIP_HUGE is set')
    def testBuildHugeOnDiskDiscardDuplicates(self):
        self._testBuild(10107530, tempfile.gettempdir(), build_kwargs = dict(discard_duplicates = True),
            gen_dupes = True)

class Id32MapperTest(IdMapperTest):
    IdMapperClass = mapped_struct.NumericId32Mapper

class ObjectIdMapperTest(IdMapperTest):
    IdMapperClass = mapped_struct.ObjectIdMapper

    def gen_values(self, *p, **kw):
        str_ = str
        for k, v in super(ObjectIdMapperTest, self).gen_values(*p, **kw):
            yield k, v
            yield str_(k), v
            yield (k,), v

    # Not supported by ObjectIdMapper
    testBuildInMemDiscardDuplicates = None
    testBuildOnDiskDiscardDuplicates = None
    testBuildHugeInMemDiscardDuplicates = None
    testBuildHugeOnDiskDiscardDuplicates = None

    # Too much memory
    testBuildHugeOnDisk = None
    testBuildHugeOnDiskReversed = None
    testBuildHugeOnDiskShuffled = None

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
            if v not in elem:  # Purposefully testing not in operator
                self.assertIn(v, elem)

    # Too much memory
    testBuildHugeInMemShuffled = None
    testBuildHugeOnDiskShuffled = None

class ApproxStringId32MultiMapperTest(ApproxStringIdMultiMapperTest):
    IdMapperClass = mapped_struct.ApproxStringId32MultiMapper

class ApproxIntegerIdMultiMapperTest(IdMapperTest):

    def gen_values(self, *p, **kw):
        for k, v in super(ApproxIntegerIdMultiMapperTest, self).gen_values(*p, **kw):
            yield int(k), v

    def _testBuild(self, N, tempdir, **gen_kwargs):
        build_kwargs = gen_kwargs.pop('build_kwargs', {})
        rv = self.IdMapperClass.build(self.gen_values(N, **gen_kwargs), tempdir = tempdir, **build_kwargs)
        for k, _ in self.gen_values(N, **gen_kwargs):
            elem = rv.get(k)
            if isinstance(elem, list):
                for e in elem:
                    self.assertTrue(isinstance(e, (int, long, numpy.uint32)))
            else:
                self.assertTrue(isinstance(elem, (int, long, numpy.uint32, numpy.uint64)))

    # Too much memory
    testBuildHugeInMemShuffled = None
    testBuildHugeOnDiskShuffled = None

class ApproxIntId32MultiMapperTest(ApproxIntegerIdMultiMapperTest):
    IdMapperClass = mapped_struct.ApproxStringId32MultiMapper

class MappedMappingTest(unittest.TestCase):
    # Reuse test values from MappedArrayTest to simplify test code
    Struct = MappedArrayTest.Struct
    TEST_VALUES = MappedArrayTest.TEST_VALUES
    TEST_KEYS = [ x*7 for x in xrange(len(TEST_VALUES)) ]
    IdMapperClass = mapped_struct.NumericIdMapper

    def _alt_k(self, k):
        return k + 1000000

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

        if not isinstance(test_values, dict):
            test_values = dict(test_values)

        # test basic attributes
        self.assertEqual(len(set(itervalues(test_values))), len(mapping))

        # test key iteration and enumeration
        self.assertEqual(set(test_values.keys()), set(mapping.keys()))
        self.assertEqual(set(iterkeys(test_values)), set(iterkeys(mapping)))

        # test lookup
        for k,reference in iteritems(test_values):
            self.assertStructEquals(reference, mapping.get(k))
        for k,reference in iteritems(test_values):
            self.assertStructEquals(reference, mapping[k])

        # test item iteration and enumeration
        for k,proxy in mapping.iteritems():
            reference = test_values[k]
            self.assertStructEquals(reference, proxy)
        for k,proxy in mapping.items():
            reference = test_values[k]
            self.assertStructEquals(reference, proxy)

    def testBuildNoIdmap(self):
        mapped = self.MappedMappingClass.build(self.test_values)
        self.assertMappingOk(mapped)

    def testBuildWithIdmap(self):
        mapped = self.MappedMappingClass.build(self.test_values, idmap = {})
        self.assertMappingOk(mapped)

    def testBuildValueDedup(self):
        test_values = self.test_values
        if isinstance(test_values, dict):
            test_values = iteritems(test_values)
        self.test_values = (
            list(test_values)
            + [(self._alt_k(k), v) for k,v in test_values]
        )
        mapped = self.MappedMappingClass.build(self.test_values, idmap = {})
        self.assertMappingOk(mapped)

    def testBuildEmpty(self):
        self.MappedMappingClass.build({})

    def testBuildFromTuples(self):
        mapped = self.MappedMappingClass.build(self.test_values)
        self.assertMappingOk(mapped)

    def testFileMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build(self.test_values, destfile = destfile, idmap = {})
            mapped = self.MappedMappingClass.map_file(destfile)
            self.assertMappingOk(mapped)

    def testZipMapping(self):
        with tempfile.NamedTemporaryFile() as destfile:
            self.MappedMappingClass.build(self.test_values, destfile = destfile, idmap = {})
            with tempfile.NamedTemporaryFile() as tempzip:
                zf = zipfile.ZipFile(tempzip, 'w')
                zf.write(destfile.name, 'bundle', zipfile.ZIP_STORED)
                zf.writestr('otherdata', 'blablabla')
                zf.close()

                tempzip.seek(0)
                zf = zipfile.ZipFile(tempzip, 'r')
                mapped = self.MappedMappingClass.map_file(zf.open('bundle'))
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
        self.assertEqual(len(set(v for k,v in test_values)), len(mapping))

        # test key iteration and enumeration
        test_keys = list(map(operator.itemgetter(0), test_values))
        self.assertEqual(set(test_keys), set(mapping.keys()))
        self.assertEqual(set(test_keys), set(iterkeys(mapping)))

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
    TEST_KEYS = list(map(lambda x: str(x).encode(), MappedMappingTest.TEST_KEYS)) + [os.urandom(65537)]
    TEST_VALUES = MappedMappingTest.TEST_VALUES + [{
        'fset' : frozenset([(1,2),(3,4),(6,7)]),
        't' : ((3,),(6,7,8),(1,7)),
        'l' : [[1],[2,3],(3,4)],
    }]

    def _alt_k(self, k):
        return k + b'_alt'

class MappedStringMultiMappingTest(MappedMultiMappingTest):
    IdMapperClass = mapped_struct.StringIdMultiMapper
    TEST_KEYS = MappedStringMappingTest.TEST_KEYS * 2
    TEST_VALUES = MappedStringMappingTest.TEST_VALUES * 2

    def _alt_k(self, k):
        return k + b'_alt'

class MappedStringMappingRepeatedValuesTest(MappedStringMappingTest):
    TEST_KEYS = MappedStringMappingTest.TEST_KEYS + list(map(b'X2_'.__add__, MappedStringMappingTest.TEST_KEYS))
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
        self.assertEqual(len(set(v for k,v in test_values)), len(mapping))

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
    TEST_KEYS = [u"%s€ ···YEAH···".encode("utf8") % (k,) for k in MappedStringMappingTest.TEST_KEYS]

class MappedString32MappingUnicodeTest(MappedString32MappingTest):
    TEST_KEYS = MappedStringMappingUnicodeTest.TEST_KEYS

class MappedStringMappingBigTest(MappedStringMappingTest):
    TEST_KEYS = [ b"%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in xrange(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in xrange(64) ]

class MappedString32MappingBigTest(MappedString32MappingTest):
    TEST_KEYS = [ b"%d%s" % (i,k) for k in MappedStringMappingTest.TEST_KEYS for i in xrange(64) ]
    TEST_VALUES = [ v for v in MappedStringMappingTest.TEST_VALUES for i in xrange(64) ]

class BsearchTest(unittest.TestCase):
    SUPPORTED_DTYPES = [ numpy.uint32, numpy.int32, numpy.uint64, numpy.int64,
        numpy.double, numpy.single, numpy.float64, numpy.float32,
        numpy.uint16, numpy.int16, numpy.uint8, numpy.int8 ]

    UNSUPPORTED_DTYPES = [ numpy.complex ]

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


class CollectionPackingTestHelpers(object):

    def pack(self, obj, buffer_size=1024):
        obj = self.COLLECTION_CLASS(obj)
        a = bytearray(buffer_size)
        self.PACKING_CLASS.pack_into(obj, a, 0)
        return self.PACKING_CLASS.unpack_from(a, 0)

    def assertPackingOk(self, obj, buffer_size=1024):
        obj = self.COLLECTION_CLASS(obj)
        c = self.pack(obj, buffer_size)
        self.assertEqual(c, obj)

class CommonCollectionPackingTest(CollectionPackingTestHelpers):

    def testCommonCollectionUnpackOffBounds(self):
        b = buffer(b"")
        self.assertRaises(Exception, self.PACKING_CLASS.unpack_from, b, 5)

    def testCommonCollectionEmpty(self):
        self.assertPackingOk([])

    def testCommonCollectionWithBytes(self):
        self.assertPackingOk([1])
        self.assertPackingOk([1, 2, 3])
        self.assertPackingOk([1, -2, 3])

    def testCommonCollectionWithShorts(self):
        self.assertPackingOk([1000])
        self.assertPackingOk([1000, 2000, 3000])
        self.assertPackingOk([1000, -2000, 3000])

    def testCommonCollectionWithInts(self):
        self.assertPackingOk([100000])
        self.assertPackingOk([100000, 200000, 300000])
        self.assertPackingOk([100000, -200000, 300000])

    def testCommonCollectionWithLongLong(self):
        self.assertPackingOk([10000000])
        self.assertPackingOk([10000000, 20000000, 30000000])
        self.assertPackingOk([10000000, -20000000, 30000000])

    def testCommonCollectionWithFloats(self):
        self.assertPackingOk([1.1])
        self.assertPackingOk([1.1, -2.2, 3.3])

    def testCommonCollectionWithObjects(self):
        self.assertPackingOk([1, 2.2, "a", frozenset([1, 2]), tuple([3, 4])])

    def testCommonCollectionContains(self):
        c = self.pack([])
        self.assertNotIn(1, c)

        c = self.pack([1])
        self.assertIn(1, c)
        self.assertNotIn(2, c)

        c = self.pack([1, 2, 3])
        self.assertIn(1, c)
        self.assertIn(2, c)
        self.assertIn(3, c)
        self.assertNotIn(4, c)
        self.assertNotIn(None, c)
        self.assertNotIn('asd', c)

class IndexedCollectionPackingTest(CollectionPackingTestHelpers):

    def assertIndexOk(self, values):
        c = self.pack(values)

        for i, v in enumerate(values):
            self.assertEqual(c[i], v)

    def testIndexedCollectionInts(self):
        self.assertIndexOk([1])
        self.assertIndexOk([1, 2, 3])

    def testIndexedCollectionFloats(self):
        self.assertIndexOk([1])
        self.assertIndexOk([1, 2, 3])

    def testIndexedCollectionObjects(self):
        self.assertIndexOk(['1', '2', '3.0', frozenset([1,2,3]), [1,2,3]])

    def testIndexedCollectionStructs(self):

        mapped_struct.mapped_object.TYPE_CODES.pop(SimpleStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop(b'}',None)

        schema = mapped_struct.Schema.from_typed_slots(SimpleStruct)
        mapped_struct.mapped_object.register_schema(SimpleStruct, schema, b'}')

        c = self.pack([SimpleStruct(a=1, b=2.0), SimpleStruct(a=2, b=None)])

        self.assertEquals(c[0].a, 1)
        self.assertEquals(c[0].b, 2.0)
        self.assertEquals(c[1].a, 2)
        self.assertEquals(c[1].b, None)

    def testIndexedCollectionError(self):
        c = self.pack([])
        self.assertRaises(IndexError, c.__getitem__, 0)

        c = self.pack([1])
        self.assertRaises(IndexError, c.__getitem__, 1)

        c = self.pack([1, 2, 3])
        self.assertRaises(IndexError, c.__getitem__, 3)

    def testIndexedCollectionIterators(self):
        c = self.pack([])
        self.assertEqual([v for v in c], [])

        c = self.pack([1])
        self.assertEqual([v for v in c], [1])

        c = self.pack([1, '2', 3.0])
        self.assertEqual([v for v in c], [1, '2', 3.0])

    def testIndexedCollectionReversed(self):
        c = self.pack([])
        self.assertEqual([v for v in c], [])

        c = self.pack([1])
        self.assertEqual([v for v in c], [1])

        c = self.pack([1, '2', 3.0])
        self.assertEqual([v for v in reversed(c)], [3.0, '2', 1])

    def testIndexedCollectionEqual(self):
        pack = self.pack
        c = pack([1, 2.0])

        self.assertNotEquals(c, None)
        self.assertNotEquals(c, dict())
        self.assertNotEquals(c, 'list')

        self.assertEquals(c, c)
        self.assertEquals(c, pack([1, 2.0]))
        self.assertEquals(c, pack([1, 2]))

        self.assertNotEquals(c, pack([]))
        self.assertNotEquals(c, pack([1]))
        self.assertNotEquals(c, pack([1, 2.0, 3]))
        self.assertNotEquals(c, pack([2, 2.0]))

class ProxiedFrozensetPackingTest(unittest.TestCase, CommonCollectionPackingTest):
    PACKING_CLASS = mapped_struct.proxied_frozenset
    COLLECTION_CLASS = frozenset

    def testContains(self):
        values = [float(x) for x in range(5)]
        c = self.pack(values)
        for val in values:
            self.assertIn(val, c)
        for val in values:
            self.assertNotIn(-val - 1, c)

    def testCmpSets(self):
        c = self.pack([1., 2., 3., 4., 5.])
        self.assertEqual(c, self.pack([5., 4., 3., 2., 1.]))
        self.assertEqual(c, set(c))
        self.assertEqual(c, frozenset(c))
        self.assertNotEqual(c, set())
        self.assertNotEqual(c, frozenset())
        c = self.pack([])
        self.assertEqual(c, set())
        self.assertEqual(c, frozenset())
        self.assertNotIn(3, c)

    def testSetOps(self):
        c = self.pack([1., 2., 3., 4., 5.])
        self.assertEqual(c.union(set([3., 6.])), frozenset([1., 2., 3., 4., 5., 6.]))
        self.assertEqual(c.intersection(set([3., 5., 9.])), frozenset([3., 5.]))
        self.assertEqual(c.difference(set([2., 4., 8.])), frozenset([1., 3., 5.]))
        self.assertEqual(c.symmetric_difference(set([1., 8.])), frozenset([2., 3., 4., 5., 8.]))
        self.assertEqual(c.union(set()), c)
        self.assertEqual(c.intersection(set()), set())
        self.assertEqual(c.difference(set()), c)

        c = self.pack([1, 101, 70])
        self.assertEqual(c & c, c)
        self.assertEqual(c | self.pack([2, 102]), self.pack([1, 2, 70, 101, 102]))
        self.assertNotIn(101, c - self.pack([101]))
        self.assertEqual(c ^ c, self.pack([]))

    def testSubsetSuperset(self):
        small = self.pack([1., 2.])
        big = self.pack([1., 2., 3., 4.])
        self.assertTrue(small < big)
        self.assertTrue(small <= big)
        self.assertFalse(small < small)
        self.assertTrue(small <= small)
        self.assertFalse(small > small)
        self.assertTrue(small >= small)
        self.assertTrue(big > small)
        self.assertTrue(big >= small)
        self.assertFalse(big > big)

        big = self.pack([1., 1.5, 2., 2.5, 3.])
        self.assertTrue(small < big)
        self.assertTrue(big > small)

        a = self.pack([1., 2.])
        b = self.pack([1., 2., 3., 4.])
        self.assertTrue(a < b)
        self.assertTrue(a <= b)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)
        self.assertNotEquals(a, b)

        a = self.pack([1., 2., 3.])
        b = self.pack([1., 2.5, 3., 4.])
        self.assertFalse(a < b)
        self.assertFalse(a <= b)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)
        self.assertNotEquals(a, b)

        a = self.pack([1., 2.5, 3.])
        b = self.pack([1., 2., 3., 4.])
        self.assertFalse(a < b)
        self.assertFalse(a <= b)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)
        self.assertNotEquals(a, b)

        a = self.pack([1., 2., 3., 10.])
        b = self.pack([1., 2., 3., 4.])
        self.assertFalse(a < b)
        self.assertFalse(a <= b)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)
        self.assertNotEquals(a, b)


    def testCompressed(self):
        small = self.pack([1, 2])
        big = self.pack([1, 2, 3, 4, 5])

        self.assertTrue(small < big)
        self.assertTrue(small <= big)
        self.assertFalse(small < small)
        self.assertTrue(big > small)
        self.assertTrue(big >= small)
        self.assertFalse(big > big)

        # Test with the higher bitmap != 0
        small = self.pack([100, 102, 104])
        big = self.pack([100, 101, 102, 103, 104, 105])
        self.assertTrue(small < big)
        self.assertTrue(small <= big)
        self.assertFalse(small < small)
        self.assertTrue(big > small)
        self.assertTrue(big >= small)
        self.assertFalse(big > big)

    def testMixed(self):
        small = self.pack([1, 2])
        big = self.pack([1., 2, 3., 4, 5.])

        self.assertTrue(small < big)
        self.assertTrue(small <= big)
        self.assertFalse(small < small)
        self.assertTrue(big > small)
        self.assertTrue(big >= small)
        self.assertFalse(big > big)

        big = self.pack([long(1), long(2), long(3)])
        self.assertTrue(small < big)
        self.assertTrue(small <= big)

    def testNonNumeric(self):
        small = self.pack(["aa", "bb", "cc"])
        big = self.pack(["aa", "ab", "bb", "bc", "cc", "cd"])

        self.assertIn("aa", small)
        self.assertNotIn("ax", small)

        self.assertTrue(small < big)
        self.assertTrue(small <= big)
        self.assertFalse(small < small)
        self.assertTrue(big > small)
        self.assertTrue(big >= small)
        self.assertFalse(big > big)

    def testForeign(self):
        obj = [2., 1, 1.5]
        small = self.pack(obj)
        self.assertTrue(small <= set(obj))
        self.assertTrue(small <= frozenset(obj))

class MappedFrozensetPackingTest(unittest.TestCase, CommonCollectionPackingTest):
    PACKING_CLASS = mapped_struct.mapped_frozenset
    COLLECTION_CLASS = frozenset

    def testIterators(self):
        c = self.pack([])
        self.assertEqual([v for v in c], [])

        c = self.pack([1])
        self.assertEqual([v for v in c], [1])

        values = [1, '2', 3.0]
        c = self.pack(values)
        self.assertEqual(frozenset([v for v in c]), frozenset(values))

    def testSingletons(self):
        a = bytearray(16)
        fs = frozenset()
        mapped_struct.mapped_frozenset.pack_into(fs, a, 0)
        self.assertIs(mapped_struct.mapped_frozenset.unpack_from(a, 0), fs)

    def testUnpackBeyondEnd(self):
        b = buffer(b"m")
        self.assertRaises(IndexError, self.PACKING_CLASS.unpack_from, b, 0)

    def testBitmapSets(self):
        a = bytearray(16)
        bitmap_values = [
            frozenset([1,3,6]),
            frozenset([10,30,60]),
            frozenset([69,99]),
            frozenset([1,5,7,38,49,67,99,105,119]),
            frozenset([119]),
        ]
        for fs in bitmap_values:
            mapped_struct.mapped_frozenset.pack_into(fs, a, 0)
            self.assertEqual(mapped_struct.mapped_frozenset.unpack_from(a, 0), fs)

class MappedListPackingTest(unittest.TestCase, CommonCollectionPackingTest, IndexedCollectionPackingTest):
    PACKING_CLASS = mapped_struct.mapped_list
    COLLECTION_CLASS = list

class MappedTuplePackingTest(unittest.TestCase, CommonCollectionPackingTest, IndexedCollectionPackingTest):
    PACKING_CLASS = mapped_struct.mapped_tuple
    COLLECTION_CLASS = tuple

class ProxiedListPackingTest(unittest.TestCase, CommonCollectionPackingTest, IndexedCollectionPackingTest):
    PACKING_CLASS = mapped_struct.proxied_list
    COLLECTION_CLASS = list

    def testProxiedListCmp(self):
        pack = self.pack

        c1 = pack([])
        c2 = pack([1, 2.0])
        c3 = pack([1, 2.0, 3])
        c4 = pack([2, 2.0])
        c5 = pack([1, 0.5, 2])

        self.assertTrue(c1 < c2)
        self.assertTrue(c1 <= c2)
        self.assertFalse(c1 > c2)
        self.assertFalse(c1 >= c2)

        self.assertTrue(c2 < c3)
        self.assertTrue(c2 <= c3)
        self.assertFalse(c2 > c3)
        self.assertFalse(c2 >= c3)

        self.assertTrue(c2 < c4)
        self.assertTrue(c2 <= c4)
        self.assertFalse(c2 > c4)
        self.assertFalse(c2 >= c4)
        self.assertFalse(c2 < c5)

        self.assertRaises(NotImplementedError, lambda: c1 < None)
        self.assertRaises(NotImplementedError, lambda: c1 < 1)

        self.assertRaises(NotImplementedError, lambda: c1 <= None)
        self.assertRaises(NotImplementedError, lambda: c1 <= 1)

        self.assertRaises(NotImplementedError, lambda: c1 > None)
        self.assertRaises(NotImplementedError, lambda: c1 > 1)

        self.assertRaises(NotImplementedError, lambda: c1 >= None)
        self.assertRaises(NotImplementedError, lambda: c1 >= 1)

    def testProxiedListSetItem(self):
        c = self.pack([1, 2, 3])
        self.assertRaises(TypeError, c.__setitem__, 0, 1)

    def testProxiedListDelItem(self):
        c = self.pack([1, 2, 3])
        self.assertRaises(TypeError, c.__delitem__, 0)

    def testProxiedListIter(self):
        l = [1, 2.0, None]
        c = self.pack(l)

        self.assertEquals(list(c), l)
        for i, x in enumerate(c):
            self.assertEquals(x, l[i])

        self.assertEquals(list(c.iter()), l)
        for i, x in enumerate(c.iter()):
            self.assertEquals(x, l[i])

        mask = numpy.zeros(len(l), numpy.uint8)
        mask[::2] = True
        self.assertEquals(list(c.iter(mask=mask)), l[::2])
        for i, x in enumerate(c.iter(mask=mask)):
            i *= 2
            self.assertEquals(x, l[i])

    def testProxiedListIterFast(self):
        mapped_struct.mapped_object.TYPE_CODES.pop(SimpleStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop(b'}',None)

        schema = mapped_struct.Schema.from_typed_slots(SimpleStruct)
        mapped_struct.mapped_object.register_schema(SimpleStruct, schema, b'}')

        l = []
        l.append(SimpleStruct(a=1, b=2.0))
        l.append(SimpleStruct(a=2, b=1.0))
        l.append(SimpleStruct(a=3, b=1.5))
        l.append(3)
        l.append([1,2,3])
        l.append(frozenset([1,2,3]))
        l.append(None)
        c = self.pack(l)

        oldx = None
        for i, x in enumerate(c.iter(proxy_into=schema.Proxy())):
            if isinstance(x, mapped_struct.BufferProxyObject):
                self.assertEquals(x.a, l[i].a)
                self.assertEquals(x.b, l[i].b)
                if oldx is not None:
                    self.assertIs(x, oldx)
                oldx = x
            else:
                self.assertEquals(x, l[i])

        oldx = None
        for i, x in enumerate(c.iter_fast()):
            if isinstance(x, mapped_struct.BufferProxyObject):
                self.assertEquals(x.a, l[i].a)
                self.assertEquals(x.b, l[i].b)
                if oldx is not None:
                    self.assertIs(x, oldx)
                oldx = x
            else:
                self.assertEquals(x, l[i])

        mask = numpy.zeros(len(c), numpy.uint8)
        mask[::2] = True
        oldx = None
        for i, x in enumerate(c.iter_fast(mask=mask)):
            i *= 2
            if isinstance(x, mapped_struct.BufferProxyObject):
                self.assertEquals(x.a, l[i].a)
                self.assertEquals(x.b, l[i].b)
                if oldx is not None:
                    self.assertIs(x, oldx)
                oldx = x
            else:
                self.assertEquals(x, l[i])

    def testProxiedListGetterFast(self):
        mapped_struct.mapped_object.TYPE_CODES.pop(SimpleStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop(b'}',None)

        schema = mapped_struct.Schema.from_typed_slots(SimpleStruct)
        mapped_struct.mapped_object.register_schema(SimpleStruct, schema, b'}')

        l = []
        l.append(SimpleStruct(a=1, b=2.0))
        l.append(SimpleStruct(a=2, b=1.0))
        l.append(SimpleStruct(a=3, b=1.5))
        l.append(3)
        l.append([1,2,3])
        l.append(frozenset([1,2,3]))
        l.append(None)
        c = self.pack(l)
        g = c.getter(fast=True)

        oldx = None
        for i in xrange(len(c)):
            x = g(i)
            if isinstance(x, mapped_struct.BufferProxyObject):
                self.assertEquals(x.a, l[i].a)
                self.assertEquals(x.b, l[i].b)
                if oldx is not None:
                    self.assertIs(x, oldx)
                oldx = x
            else:
                self.assertEquals(x, l[i])

        oldx = None
        for i in xrange(len(c)):
            x = g(i)
            if isinstance(x, mapped_struct.BufferProxyObject):
                self.assertEquals(x.a, l[i].a)
                self.assertEquals(x.b, l[i].b)
                if oldx is not None:
                    self.assertIs(x, oldx)
                oldx = x
            else:
                self.assertEquals(x, l[i])

    def testProxiedListStr(self):
        c = self.pack([1, 2.0, None])
        self.assertEquals(str(c), "[1,2.0,None]")

    def testProxiedListRepr(self):
        c = self.pack([1, 2.0, None])
        self.assertEquals(repr(c), "proxied_list([1,2.0,None])")

    def testProxiedListSpecificEqual(self):
        self.assertEquals(self.pack([1, 2.0]), (1, 2.0))
        self.assertEquals(self.pack([1, None, 2.0]), (1, None, 2.0))

    def testProxiedListSlice(self):
        orig = [(str(i) if (i % 2) == 0 else i) for i in range(20)]
        obj = self.pack(orig)
        huge = 1 << 40

        self.assertEquals(obj[1:], orig[1:])
        self.assertEquals(obj[:-1], orig[:-1])
        self.assertEquals(obj[2:4], orig[2:4])
        self.assertEquals(obj[1:-1:2], orig[1:-1:2])
        self.assertEquals(obj[::1], orig[::1])
        self.assertEquals(obj[::2], orig[::2])
        self.assertEquals(obj[::-1], orig[::-1])
        self.assertEquals(obj[1::-1], orig[1::-1])
        self.assertEquals(obj[2:][:5], orig[2:][:5])
        self.assertEquals(obj[::3][::2], orig[::3][::2])
        self.assertEquals(obj[1:10:2][2::5], orig[1:10:2][2::5])
        self.assertEquals(obj[10:1:-3][2::2], orig[10:1:-3][2::2])
        self.assertEquals(obj[:], orig[:])
        self.assertEquals(obj[::], orig[::])
        self.assertEquals(obj[huge:huge], orig[huge:huge])
        self.assertEquals(obj[huge:huge:huge], orig[huge:huge:huge])

    def testProxiedListSliceOutOfBounds(self):
        orig = range(10)
        xlen = len(orig)
        obj = self.pack(range(10))

        self.assertRaises(IndexError, lambda: obj[1:][xlen - 1])
        self.assertRaises(IndexError, lambda: obj[-1::][1])
        self.assertRaises(IndexError, lambda: obj[2:4:2][xlen // 2 - 2])
        self.assertRaises(IndexError, lambda: obj[:2:-2][xlen // 2])
        self.assertRaises(TypeError, lambda: obj[1:'a'])

    def testProxiedListSliceNotComparable(self):
        obj = self.pack(range(3))
        self.assertRaises(NotImplementedError, lambda: obj < "hello")
        self.assertRaises(NotImplementedError, lambda: obj >= 42)
        self.assertNotEquals(obj, None)
        self.assertNotEquals(obj, "123456")

    def testProxiedListNotEqual(self):
        obj = self.pack([1, 2, 3, 4, 5])
        self.assertNotEqual(obj, (1, 2, 3, 4))
        self.assertNotEqual(obj[1:], (2, 3, 4))
        self.assertNotEqual(obj[:-1], (1, 2, 3, 4, 5))
        self.assertNotEqual(obj[2:4], (2, 3, 4))


class ProxiedTuplePackingTest(unittest.TestCase, CommonCollectionPackingTest, IndexedCollectionPackingTest):
    PACKING_CLASS = mapped_struct.proxied_tuple
    COLLECTION_CLASS = tuple

    def testProxiedTupleHash(self):
        p = self.pack([1, 2.0])
        t = (1, 2.0)

        self.assertTrue(hash(p) == hash(t))
        self.assertTrue(p == t)

    def testProxiedTupleCmp(self):
        pack = self.pack

        c1 = pack([])
        c2 = pack([1, 2.0])
        c3 = pack([1, 2.0, 3])
        c4 = pack([2, 2.0])

        self.assertTrue(c1 < c2)
        self.assertTrue(c1 <= c2)
        self.assertFalse(c1 > c2)
        self.assertFalse(c1 >= c2)

        self.assertTrue(c2 < c3)
        self.assertTrue(c2 <= c3)
        self.assertFalse(c2 > c3)
        self.assertFalse(c2 >= c3)

        self.assertTrue(c2 < c4)
        self.assertTrue(c2 <= c4)
        self.assertFalse(c2 > c4)
        self.assertFalse(c2 >= c4)

        self.assertRaises(NotImplementedError, lambda: c1 < None)
        self.assertRaises(NotImplementedError, lambda: c1 < 1)

        self.assertRaises(NotImplementedError, lambda: c1 <= None)
        self.assertRaises(NotImplementedError, lambda: c1 <= 1)

        self.assertRaises(NotImplementedError, lambda: c1 > None)
        self.assertRaises(NotImplementedError, lambda: c1 > 1)

        self.assertRaises(NotImplementedError, lambda: c1 >= None)
        self.assertRaises(NotImplementedError, lambda: c1 >= 1)


    def testProxiedTupleSpecificEqual(self):
        self.assertEquals(self.pack([1, 2.0]), [1, 2.0])

    def testProxiedTupleStr(self):
        c = self.pack([1, 2.0])
        self.assertEquals(str(c), "(1,2.0)")

    def testProxiedTupleRepr(self):
        c = self.pack([1, 2.0])
        self.assertEquals(repr(c), "proxied_tuple((1,2.0))")

def tuple_cmp(a, b):
    for ea, eb in zip(a,b):
        if key_cmp(ea, eb):
            return key_cmp(ea, eb)
    return 0

def key_cmp(a, b):
    if type(a) != type(b):
        return cmp(type(a).__name__, type(b).__name__)
    if isinstance(a, tuple):
        return tuple_cmp(a, b)
    return cmp(a, b)

class DictPackingCommonTest(object):

    def assertUnsortedEquals(self, a, b):
        if six.PY3:
            return sorted(a, key=functools.cmp_to_key(key_cmp)) == sorted(b, key=functools.cmp_to_key(key_cmp))
        return sorted(a) == sorted(b)

    def testMappedDictPrimitives(self):
        for d in self.TEST_DICTS:
            self.assertPackingOk(d)

    def testMappedDictStructs(self):
        mapped_struct.mapped_object.TYPE_CODES.pop(SimpleStruct,None)
        mapped_struct.mapped_object.OBJ_PACKERS.pop(b'}',None)

        schema = mapped_struct.Schema.from_typed_slots(SimpleStruct)
        mapped_struct.mapped_object.register_schema(SimpleStruct, schema, b'}')

        d = {
            'a': SimpleStruct(a=1, b=2.0),
            'b': SimpleStruct(a=2, b=None),
            None: SimpleStruct(a=3, b=1.0),
            (): SimpleStruct(a=4, b=1.5),
        }
        c = self.pack(d)

        self.assertEquals(c['a'].a, 1)
        self.assertEquals(c['a'].b, 2.0)
        self.assertEquals(c['b'].a, 2)
        self.assertEquals(c['b'].b, None)
        self.assertEquals(c[None].a, 3)
        self.assertEquals(c[None].b, 1.0)
        self.assertEquals(c[()].a, 4)
        self.assertEquals(c[()].b, 1.5)

    def testMappedDictKeys(self):
        for d in self.TEST_DICTS:
            p = self.pack(d)
            self.assertUnsortedEquals(d.keys(), p.keys())

    def testMappedDictValues(self):
        for d in self.TEST_DICTS:
            p = self.pack(d)
            self.assertUnsortedEquals(d.values(), p.values())

    def testMappedDictIterator(self):
        for d in self.TEST_DICTS:
            p = self.pack(d)
            self.assertUnsortedEquals([v for v in d], [v for v in p])

    def testMappedDictIterItems(self):
        for d in self.TEST_DICTS:
            p = self.pack(d)
            self.assertUnsortedEquals([v for v in iteritems(d)], [v for v in iteritems(p)])


class MappedDictPackingTest(unittest.TestCase, CollectionPackingTestHelpers, DictPackingCommonTest):
    PACKING_CLASS = mapped_struct.mapped_dict
    COLLECTION_CLASS = dict

    TEST_DICTS = [
        {},
        {'a': 'a2', 'b': 'b2', 'c': 'c2'},
        {1: 10, 2: 20, 3: 30},
        {1.0: 10.0, 2.0: 2.2, 3.0: 3.3},
        {frozenset([1]): frozenset(['a']), frozenset([2]): frozenset(['b'])},
        {'a': 1, 1: 'a', frozenset(): 1.0, (1, 2): 80000 },
        {None: 3},
        {(1,2,3): 4, (-1,3): 7},
        {frozenset([1,2,3]): 4, frozenset([-1,3]): 7},
    ]

class ProxiedDictPackingTest(unittest.TestCase, CollectionPackingTestHelpers, DictPackingCommonTest):
    PACKING_CLASS = mapped_struct.proxied_dict
    COLLECTION_CLASS = dict

    TEST_DICTS = [
        {},
        {'a': 'a2', 'b': 'b2', 'c': 'c2'},
        {1: 10, 2: 20, 3: 30},
        {'a': frozenset(), 'b': (1, 2), 'c': 1.0, 'd': [1, 2], 'e': dict(a=1) },
        {0: 42, 'a1_@!': 69, -3.5: 'uhhhh', (1, 2, 3): "four-five-six"},
        {frozenset([1, 2]) : 97.9},
        {1.0: "test floats equivalent to integers"},
        {None: 3},
        {(1,2,3): 4, (-1,3): 7},
        {frozenset([1,2,3]): 4, frozenset([-1,3]): 7},
        {float("inf"): 0, float("-inf"): 1, -0.0001: 2, -123456.: 3}
    ]

class StableHashTest(unittest.TestCase):

    def assertHashesOK(self, values_hashes):
        for i, (value, hval) in enumerate(values_hashes):
            hv2 = mapped_struct._stable_hash(value)
            self.assertTrue(isinstance(hv2, (int, long)))
            self.assertEqual(hv2, hval)

    def testHashIntegersFixed(self):
        values_hashes = (
            (1, 1),
            (1033, 1033),
            (8620194, 8620194),
            (20913041029, 20913041029),
            (66210231110, 66210231110),
            (752, 752),
            (-1, long(18446744073709551615)),
            (1 << 100, 1),
            (-(1 << 100), 1),
            (1, 1),
            (-1, long(18446744073709551615)),
        )
        self.assertHashesOK(values_hashes)

    def testHashFloatsFixed(self):
        values_hashes = (
            (1., 1),
            (4096., 4096),
            (107203., 107203),
            (91024215., 91024215),
            (13328914., 13328914),
            (-1., long(18446744073709551615)),
            (1e+50, long(150463276902340)),
            (1e-50, long(263280728297184)),
            (2.001, long(140807857099441)),
            (-2.001, long(18446603265852452175)),
            (float('inf'), long(281474976710655)),
            (float('-inf'), long(18446462598732840961)),
            (float('nan'), long(562949953421310)),
        )
        self.assertHashesOK(values_hashes)

    def testHashStringsFixed(self):
        values_hashes = (
            ("abcdef", long(18053520794346263629)),
            ("123456789", long(10139926970967174787)),
            ("!@#%&$", long(15648343848775299486)),
        )
        self.assertHashesOK(values_hashes)

    def testHashSequenceFixed(self):
        values_hashes = (
            ((1, "abc", -2.3), long(2059039662577021167)),
            # FIXME this doesnt work, the frozensets have differente order in py3 vs py2
            # (frozenset([1.2, 4.5, 0.12]), 4015877951310865576),
        )
        self.assertHashesOK(values_hashes)

class MappedDatetimePackingTest(unittest.TestCase):

    TEST_VALUE_NOW = datetime.now()
    TEST_VALUE_OLD = datetime(1900, 1, 1, 1, 2, 3, 400)

    def testPack(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_datetime = mapped_struct.mapped_datetime
        size = mapped_datetime.pack_into(now, buf, 0)
        self.assertTrue(size > 0)

        unpacked_now = mapped_datetime.unpack_from(buf, 0)
        self.assertEquals(now, unpacked_now)

        # With offset
        mapped_datetime = mapped_struct.mapped_datetime
        self.assertEquals(mapped_datetime.pack_into(now, buf, 2), size + 2)

        unpacked_now = mapped_datetime.unpack_from(buf, 2)
        self.assertEquals(now, unpacked_now)

    def testPackNonUTC(self):
        with set_tz('GMT+3'):
            self.testPack()

    def testPackUTC(self):
        with set_tz('UTC'):
            self.testPack()

    def testPackUnpackAcrossTZ(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_datetime = mapped_struct.mapped_datetime

        with set_tz('GMT-3'):
            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('UTC'):
            # Will return local time
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(now, unpacked_now + timedelta(hours=3))

            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT-3'):
            # Will return local time
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(now, unpacked_now + timedelta(hours=-3))

    def testPackUnpackAcrossTZAssumeLocal(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_datetime = mapped_struct.mapped_datetime_local

        with set_tz('GMT-3'):
            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT'):
            # Since internal is UTC and assumed is local, return should differ from input
            # to reflect TZ changes. Returned will have local tzinfo
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'GMT')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=0))
            self.assertNotEquals(now, unpacked_now.replace(tzinfo=None))

            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT-3'):
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'GMT')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=3))
            self.assertNotEquals(now, unpacked_now.replace(tzinfo=None))

    def testPackUnpackAcrossTZAssumeUTC(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_datetime = mapped_struct.mapped_datetime_utc

        with set_tz('GMT-3'):
            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT'):
            # Since internal is UTC and assumed is UTC, return should exactly match input
            # Except returned will have UTC tzinfo
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=0))
            self.assertEquals(now, unpacked_now.replace(tzinfo=None))

            size = mapped_datetime.pack_into(now, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT-3'):
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=0))
            self.assertEquals(now, unpacked_now.replace(tzinfo=None))

    def testPackUnpackExplicitTZAcrossTZ(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_datetime = mapped_struct.mapped_datetime_tz

        #NOTE: set_tz works in posix offset where we use inverted offset
        #  For posix, ART = GMT+3, but we format tz as ART = GMT-3
        #  which is how the default dateutil.tz.tzstr interprets them

        with set_tz('GMT-3'):
            reftz = dateutil.tz.tzlocal()
            refnow = now.replace(tzinfo=reftz)
            size = mapped_datetime.pack_into(refnow, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT'):
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC+3')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=3))
            self.assertEquals(refnow, unpacked_now.astimezone(reftz))

            reftz = dateutil.tz.tzlocal()
            refnow = now.replace(tzinfo=reftz)
            size = mapped_datetime.pack_into(refnow, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT+3'):
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=0))
            self.assertEquals(refnow, unpacked_now.astimezone(reftz))

            reftz = dateutil.tz.tzlocal()
            refnow = now.replace(tzinfo=reftz)
            size = mapped_datetime.pack_into(refnow, buf, 0)
            self.assertTrue(size > 0)

        with set_tz('GMT+2'):
            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC-3')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=-3))
            self.assertEquals(refnow, unpacked_now.astimezone(reftz))

            reftz = dateutil.tz.tzutc()
            refnow = now.replace(tzinfo=reftz)
            size = mapped_datetime.pack_into(refnow, buf, 0)
            self.assertTrue(size > 0)

            unpacked_now = mapped_datetime.unpack_from(buf, 0)
            self.assertEquals(unpacked_now.tzname(), 'UTC')
            self.assertEquals(unpacked_now.utcoffset(), timedelta(hours=0))
            self.assertEquals(refnow, unpacked_now.astimezone(reftz))

        REFDATES = [
            datetime(2023, 1, 10, 11, 0, 0),
            datetime(2023, 1, 10, 15, 10, 0),
            datetime(1900, 1, 10, 11, 0, 0),
            datetime(2060, 1, 10, 11, 0, 0),
        ]
        REFTZ = ['GMT-5', 'GMT-3', 'GMT+0', 'GMT+3', 'GMT+5']
        for systz in REFTZ:
            with set_tz(systz):
                for refdate in REFDATES:
                    for tznamein in REFTZ:
                        intz = dateutil.tz.tzstr(tznamein)
                        refdate = refdate.replace(tzinfo=intz)
                        size = mapped_datetime.pack_into(refdate, buf, 0)
                        self.assertTrue(size > 0)

                        unpacked_date = mapped_datetime.unpack_from(buf, 0)
                        self.assertEquals(
                            refdate, unpacked_date,
                            "%r != %r (sys tz:%r val tz=%r)" % (refdate, unpacked_date, systz, tznamein))
                        self.assertEquals(unpacked_date.tzname(), tznamein.replace('GMT', 'UTC').replace('+0', ''))
                        self.assertEquals(unpacked_date.utcoffset(), intz.utcoffset(refdate))

    def testPackOldDate(self):
        buf = bytearray(12)

        mapped_datetime = mapped_struct.mapped_datetime
        size = mapped_datetime.pack_into(self.TEST_VALUE_OLD, buf, 0)
        self.assertTrue(size > 0)

        unpacked_now = mapped_datetime.unpack_from(buf, 0)
        self.assertEquals(self.TEST_VALUE_OLD, unpacked_now)

class MappedDatePackingTest(unittest.TestCase):

    TEST_VALUE_NOW = date.today()
    TEST_VALUE_OLD = date(1900, 1, 1)

    def testPack(self):
        buf = bytearray(12)
        now = self.TEST_VALUE_NOW

        mapped_date = mapped_struct.mapped_date

        with set_tz("UTC"):
            size = mapped_date.pack_into(now, buf, 0)

        self.assertTrue(size > 0)

        with set_tz("America/Argentina/Buenos_Aires"):
            unpacked_now = mapped_date.unpack_from(buf, 0)

        self.assertEquals(now, unpacked_now)

        # With offset
        with set_tz("America/Argentina/Buenos_Aires"):
            self.assertEquals(mapped_date.pack_into(now, buf, 2), size + 2)

        with set_tz("UTC"):
            unpacked_now = mapped_date.unpack_from(buf, 2)

        self.assertEquals(now, unpacked_now)

    def testPackOldDate(self):
        buf = bytearray(12)

        mapped_date = mapped_struct.mapped_date

        with set_tz("UTC"):
            size = mapped_date.pack_into(self.TEST_VALUE_OLD, buf, 0)

        self.assertTrue(size > 0)

        with set_tz("America/Argentina/Buenos_Aires"):
            unpacked_now = mapped_date.unpack_from(buf, 0)

        self.assertEquals(self.TEST_VALUE_OLD, unpacked_now)

class MappedDecimalPackingTest(unittest.TestCase):

    TEST_CASES = [0, 100, -100, 123.456, -123.456]

    def assertPackOk(self, num):
        buf = bytearray(128)

        mapped_decimal = mapped_struct.mapped_decimal
        size = mapped_decimal.pack_into(num, buf, 0)
        self.assertTrue(size > 0)

        unpacked_num = mapped_decimal.unpack_from(buf, 0)
        self.assertEquals(str(num), str(unpacked_num))

    def testDecimal(self):
        for case in self.TEST_CASES:
            self.assertPackOk(Decimal(case))

    def testCDecimal(self):
        for case in self.TEST_CASES:
            self.assertPackOk(cDecimal(case))

class ProxiedBufferPackingTest(unittest.TestCase):

    def assertPackUnpackOk(self, obj, offs):
        proxied_buffer = mapped_struct.proxied_buffer
        buf = bytearray(64)

        new_offs = proxied_buffer.pack_into(obj, buf, offs)
        self.assertEquals(new_offs, offs + len(obj) + 8) # obj.size + ulong.size

        unpacked_obj = proxied_buffer.unpack_from(buf, offs)
        self.assertEquals(buffer(obj), unpacked_obj)

    def testBufferPackUnpack(self):
        obj = buffer(bytearray(xrange(100)))
        self.assertPackUnpackOk(obj, 0)

    def testBufferPackUnpackWithOffset(self):
        obj = buffer(bytearray(xrange(100)))
        self.assertPackUnpackOk(obj, 10)

    def testBytesPackUnpack(self):
        obj = bytes(bytearray(xrange(100)))
        self.assertPackUnpackOk(obj, 0)

    def testBytesPackUnpackWithOffset(self):
        obj = bytes(bytearray(xrange(100)))
        self.assertPackUnpackOk(obj, 10)

    def testBytearrayPackUnpack(self):
        obj = bytearray(xrange(100))
        self.assertPackUnpackOk(obj, 0)

    def testBytearrayPackUnpackWithOffset(self):
        obj = bytearray(xrange(100))
        self.assertPackUnpackOk(obj, 10)

class ProxiedNDArrayPackingTest(unittest.TestCase):

    def assertPackUnpackOk(self, value, dtype = None, offs = 0):
        proxied_ndarray = mapped_struct.proxied_ndarray
        buf = bytearray(1024)

        obj = np.array(value, dtype)
        proxied_ndarray.pack_into(obj, buf, offs)

        unpacked_obj = proxied_ndarray.unpack_from(buf, offs)

        self.assertTrue(np.array_equal(obj, unpacked_obj))

    def testPackUnpackOk(self):

        self.assertPackUnpackOk([])
        self.assertPackUnpackOk([1])
        self.assertPackUnpackOk([1,2,3])
        self.assertPackUnpackOk(["a", "", "abc"])
        self.assertPackUnpackOk([("a", 1), ("b", 2)])
        self.assertPackUnpackOk([("a", 1), ("b", "2")])

    def testPackUnpackStructsOk(self):
        self.assertPackUnpackOk(
            [('Rex', 9, 81.0), ('Fido', 3, 27.5)],
            dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')]
        )

    def testMakeDtypeParams(self):

        TEST_VALUES = [
            'i',
            [('a', 'l'), ('b', 'i')],
            [('a', 'l'), ('b', [('c', 'i')])]
        ]
        make_dtype_params = mapped_struct.proxied_ndarray._make_dtype_params

        for value in TEST_VALUES:
            dtype = np.dtype(value)
            dtype_list = make_dtype_params(dtype)
            self.assertEquals(dtype, np.dtype(dtype_list))

class BehavioralStruct(object):
    __slot_types__ = {
        'a' : int,
        'b' : float,
    }
    def somefunc(self):
        return 'someresult'

class CustomBasesTest(unittest.TestCase):
    Struct = BehavioralStruct

    def setUp(self):
        self.schema = mapped_struct.Schema.from_typed_slots(self.Struct)
        self.schema.set_proxy_bases((BehavioralStruct,))

    def testBehavior(self):
        x = self.Struct()
        dx = self.schema.unpack(self.schema.pack(x))
        self.assertIsInstance(dx, BehavioralStruct)
        self.assertEqual(dx.somefunc(), 'someresult')

class WriteableMappingTest(unittest.TestCase):
    class Numeric(object):
        __slot_types__ = {
            'Bx': mapped_struct.ubyte,
            'bx': mapped_struct.byte,
            'Hx': mapped_struct.uint16,
            'hx': mapped_struct.int16,
            'Ix': mapped_struct.uint32,
            'ix': mapped_struct.int32,
            'Qx': mapped_struct.uint64,
            'qx': mapped_struct.int64,
            'dx': mapped_struct.float64,
            'fx': mapped_struct.float32
        }
        __slots__ = __slot_types__.keys()
        __schema__ = mapped_struct.Schema.from_typed_slots(__slot_types__)

        def __init__(self):
            self.Bx = self.bx = 0
            self.Hx = self.hx = 0
            self.Ix = self.ix = 0
            self.Qx = self.qx = 0
            self.dx = self.fx = 0

    def setUp(self):
        self.schema = WriteableMappingTest.Numeric.__schema__
        class MappedNumericClass(mapped_struct.MappedArrayProxyBase):
            schema = self.schema
        self.mapped_class = MappedNumericClass
        self.single_value = WriteableMappingTest.Numeric()

    def _assert_load_store(self, proxy):
        proxy.Bx = 100
        proxy.bx = -100
        proxy.Hx = 1000
        proxy.hx = -1000
        proxy.Ix = 100000
        proxy.ix = -100000
        proxy.Qx = long(10000000000)
        proxy.qx = long(-10000000000)
        proxy.dx = 1.5
        proxy.fx = -1.5
        self.assertAlmostEqual(0.0, proxy.Bx + proxy.bx + proxy.Hx + proxy.hx +
            proxy.Ix + proxy.ix + proxy.Qx + proxy.qx + proxy.dx + proxy.fx)

    def testWritableBuffer(self):
        n = self.single_value
        buf = bytearray(1000)
        self.schema.pack_into(n, buf, 0)
        proxy = self.schema.unpack_from(buf, 0)
        self._assert_load_store(proxy)

    def testWriteableFile(self):
        values = [self.single_value]
        with tempfile.NamedTemporaryFile() as destfile:
            self.mapped_class.build(values, destfile = destfile, idmap = {})
            lst = self.mapped_class.map_file(destfile, read_only = False)
            for proxy in lst:
                self._assert_load_store(proxy)

    def testWriteableZip(self):
        values = [self.single_value]
        with tempfile.NamedTemporaryFile() as destfile:
            self.mapped_class.build(values, destfile = destfile, idmap = {})
            with tempfile.NamedTemporaryFile() as tempzip:
                zf = zipfile.ZipFile(tempzip, 'w')
                zf.write(destfile.name, 'bundle', zipfile.ZIP_STORED)
                zf.writestr('otherdata', 'blablabla')
                zf.close()

                tempzip.seek(0)
                zf = zipfile.ZipFile(tempzip, 'r')
                proxy = self.mapped_class.map_file(zf.open('bundle'), read_only = False)[0]
                self._assert_load_store(proxy)

    def testWriteableCasSuccess(self):
        n = self.single_value
        buf = bytearray(1000)
        self.schema.pack_into(n, buf, 0)
        proxy = self.schema.unpack_from(buf, 0)

        # We test not only that the member matches the expected value,
        # but also that adjacent members weren't modified.

        self.assertTrue(type(proxy).Bx.cas(proxy, 0, 1))
        self.assertEqual(proxy.Bx, 1)
        self.assertEqual(proxy.bx, 0)

        self.assertTrue(type(proxy).bx.cas(proxy, 0, -1))
        self.assertEqual(proxy.bx, -1)
        self.assertEqual(proxy.Bx + proxy.bx, 0)

        self.assertTrue(type(proxy).Hx.cas(proxy, 0, 1))
        self.assertEqual(proxy.Hx, 1)
        self.assertEqual(proxy.hx, 0)

        self.assertTrue(type(proxy).hx.cas(proxy, 0, -1))
        self.assertEqual(proxy.Hx + proxy.hx, 0)

        self.assertTrue(type(proxy).Ix.cas(proxy, 0, 1))
        self.assertEqual(proxy.Ix, 1)
        self.assertEqual(proxy.ix, 0)

        self.assertTrue(type(proxy).ix.cas(proxy, 0, -1))
        self.assertEqual(proxy.Ix + proxy.ix, 0)

        self.assertTrue(type(proxy).Qx.cas(proxy, 0, 1))
        self.assertEqual(proxy.Qx, 1)
        self.assertEqual(proxy.qx, 0)

        self.assertTrue(type(proxy).qx.cas(proxy, 0, -1))
        self.assertEqual(proxy.Qx + proxy.qx, 0)

        self.assertTrue(type(proxy).dx.cas(proxy, 0.0, 1.0))
        self.assertAlmostEqual(proxy.dx, 1.0)
        self.assertAlmostEqual(proxy.fx, 0.0)

        self.assertTrue(type(proxy).fx.cas(proxy, 0.0, -1.0))
        self.assertAlmostEqual(proxy.dx + proxy.fx, 0.0)

    def testWriteableCasFail(self):
        n = self.single_value
        buf = bytearray(1000)
        self.schema.pack_into(n, buf, 0)
        proxy = self.schema.unpack_from(buf, 0)

        self.assertFalse(type(proxy).Bx.cas(proxy, 1, 2))
        self.assertEqual(proxy.Bx, 0)

        self.assertFalse(type(proxy).bx.cas(proxy, 1, 2))
        self.assertEqual(proxy.bx, 0)

        self.assertFalse(type(proxy).Hx.cas(proxy, 1, 2))
        self.assertEqual(proxy.Hx, 0)

        self.assertFalse(type(proxy).hx.cas(proxy, 1, 2))
        self.assertEqual(proxy.hx, 0)

        self.assertFalse(type(proxy).Ix.cas(proxy, 1, 2))
        self.assertEqual(proxy.Ix, 0)

        self.assertFalse(type(proxy).ix.cas(proxy, 1, 2))
        self.assertEqual(proxy.ix, 0)

        self.assertFalse(type(proxy).Qx.cas(proxy, 1, 2))
        self.assertEqual(proxy.Qx, 0)

        self.assertFalse(type(proxy).qx.cas(proxy, 1, 2))
        self.assertEqual(proxy.qx, 0)

        self.assertFalse(type(proxy).dx.cas(proxy, 1.0, 2.0))
        self.assertAlmostEqual(proxy.dx, 0.0)

        self.assertFalse(type(proxy).fx.cas(proxy, 1.0, 2.0))
        self.assertAlmostEqual(proxy.fx, 0.0)

    def testWriteableAdd(self):
        n = self.single_value
        buf = bytearray(1000)
        self.schema.pack_into(n, buf, 0)
        proxy = self.schema.unpack_from(buf, 0)

        # We test not only that the member matches the expected value,
        # but also that adjacent members weren't modified.
        type(proxy).Bx.add(proxy, 1)
        self.assertEqual(proxy.Bx, 1)
        self.assertEqual(proxy.bx, 0)

        type(proxy).bx.add(proxy, -1)
        self.assertEqual(proxy.bx, -1)
        self.assertEqual(proxy.Bx + proxy.bx, 0)

        type(proxy).Hx.add(proxy, 1)
        self.assertEqual(proxy.Hx, 1)
        self.assertEqual(proxy.hx, 0)

        type(proxy).hx.add(proxy, -1)
        self.assertEqual(proxy.Hx + proxy.hx, 0)

        type(proxy).Ix.add(proxy, 1)
        self.assertEqual(proxy.Ix, 1)
        self.assertEqual(proxy.ix, 0)

        type(proxy).ix.add(proxy, -1)
        self.assertEqual(proxy.Ix + proxy.ix, 0)

        type(proxy).Qx.add(proxy, 1)
        self.assertEqual(proxy.Qx, 1)
        self.assertEqual(proxy.qx, 0)

        type(proxy).qx.add(proxy, -1)
        self.assertEqual(proxy.Qx + proxy.qx, 0)

        type(proxy).dx.add(proxy, 1.0)
        self.assertAlmostEqual(proxy.dx, 1.0)
        self.assertAlmostEqual(proxy.fx, 0.0)

        type(proxy).fx.add(proxy, -1.0)
        self.assertAlmostEqual(proxy.dx + proxy.fx, 0.0)
