import unittest
import rivapy.instruments as instruments
from rivapy.marketdata.factory import _factory, create


class FactoryTestsMeta(type):

    def __new__(cls, name, bases, attrs):
        for k, v in _factory().items():
            attrs["test_%s" % k] = cls.gen(v)
        return super(FactoryTestsMeta, cls).__new__(cls, name, bases, attrs)

    @classmethod
    def gen(cls, target_class):
        # Return a testcase that tests the creation of an instrument from factory
        def fn(self):
            try:
                mkt = target_class._create_sample(4, seed=42)
                for i in range(len(mkt)):
                    b = create(mkt[i].to_dict())
                    self.assertEqual(b.hash(), mkt[i].hash())
            except AttributeError as e:
                self.assertEqual(0, 1, msg="_create_sample not implemented for class " + target_class.__name__)

        return fn


class MarketDataTests(unittest.TestCase, metaclass=FactoryTestsMeta):
    pass


if __name__ == "__main__":
    unittest.main()
