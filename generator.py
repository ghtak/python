
def gene():
    for i in range(10):
        yield i + 1

for value in gene():
    print(value)


def gene1():
    for i in gene2():
        yield i

def gene2():
    for i in range(10):
        yield i

for value in gene1():
    print(value)


def gene1_1():
    yield from gene2_1()

def gene2_1():
    yield from range(10)


for value in gene1_1():
    print(value)


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
        yield i + 1


holder = gene2()
isinstance(holder, GeneratorHolder)

for v in holder:
    print(v)