import unittest


#
def gene():
    for i in range(10):
        yield i


# --
def gene1():
    for i in gene2():
        yield i


def gene2():
    for i in range(10):
        yield i


# --
def gene1_1():
    yield from gene2_1()


def gene2_1():
    yield from range(10)


# --
class GeneratorHolder:
    def __init__(self, generator) -> None:
        self.generator = generator

    def __next__(self):
        return next(self.generator)

    def __iter__(self):
        return self


def generator_hold(func, *args, **kwargs):
    def f(*args, **kwargs):
        return GeneratorHolder(func(*args, **kwargs))

    return f


@generator_hold
def gene2():
    for i in range(10):
        yield i


class TestGeneDeco(unittest.TestCase):
    def test_Holder(self):
        holder = gene2()
        self.assertTrue(isinstance(holder, GeneratorHolder))

        expected = [i for i in range(10)]
        self.assertListEqual(expected, [i for i in gene()])
        self.assertListEqual(expected, [i for i in gene1()])
        self.assertListEqual(expected, [i for i in gene1_1()])
        self.assertListEqual(expected, [i for i in holder])


if __name__ == '__main__':
    unittest.main()
