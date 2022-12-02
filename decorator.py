import unittest

from contextlib import contextmanager


@contextmanager
def managed_resource():
    res = 81
    yield res
    close_res = res


class SimpleCtxMgr:
    def __init__(self, gene) -> None:
        self.gene = gene

    def __enter__(self):
        return next(self.gene)

    def __exit__(self, type, value, traceback):
        try:
            next(self.gene)
        except StopIteration:
            return True


def simplectxmgr(func, *args, **kwargs):
    def f(*args, **kwargs):
        return SimpleCtxMgr(func(*args, **kwargs))
    return f


@simplectxmgr
def managed_resource2():
    res = 81
    yield res
    close_res = res


'''
class GeneratorContextManager(object):

   def __init__(self, gen):
       self.gen = gen

   def __enter__(self):
       try:
           return self.gen.next()
       except StopIteration:
           raise RuntimeError("generator didn't yield")

   def __exit__(self, type, value, traceback):
       if type is None:
           try:
               self.gen.next()
           except StopIteration:
               return
           else:
               raise RuntimeError("generator didn't stop")
       else:
           try:
               self.gen.throw(type, value, traceback)
               raise RuntimeError("generator didn't stop after throw()")
           except StopIteration:
               return True
           except:
               # only re-raise if it's *not* the exception that was
               # passed to throw(), because __exit__() must not raise
               # an exception unless __exit__() itself failed.  But
               # throw() has to raise the exception to signal
               # propagation, so this fixes the impedance mismatch
               # between the throw() protocol and the __exit__()
               # protocol.
               #
               if sys.exc_info()[1] is not value:
                   raise

def contextmanager(func):
   def helper(*args, **kwds):
       return GeneratorContextManager(func(*args, **kwds))
   return helper
'''


class TesteDeco(unittest.TestCase):
    def test_Deco(self):
        with managed_resource() as v:
            self.assertEqual(v, 81)

        with managed_resource2() as v:
            self.assertEqual(v, 81)


if __name__ == '__main__':
    unittest.main()
