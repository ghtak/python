import unittest


class Iterable:
    def __init__(self,
                 max_iter=5,
                 start_value=0):
        self.max_iter = max_iter
        self.start_value = start_value

    def __iter__(self):
        return Iterator(self.max_iter,
                        self.start_value)


class Iterator:
    def __init__(self,
                 max_iter,
                 start_value):
        self.value = start_value
        self.max_iter = max_iter

    def __next__(self):
        old_v = self.value
        if old_v >= self.max_iter:
            raise StopIteration
        self.value += 1
        return old_v


class TestIter(unittest.TestCase):
    def setUp(self):
        self.max_iter = 10

    def test_iter(self):
        it = iter(Iterable(self.max_iter))
        for i in range(0, self.max_iter):
            self.assertEqual(i, next(it))
        self.assertRaises(StopIteration, next, it)

    def test_iter1(self):
        for i, value in enumerate(Iterable(self.max_iter, 1)):
            self.assertEqual(i, value - 1)


if __name__ == '__main__':
    unittest.main()
