import socket
import enum
import select


class OpTypes(enum.Enum):
    READ_OP = enum.auto()
    WRITE_OP = enum.auto()


class AwaitOp:
    def __init__(self, op, fd) -> None:
        self.op = op
        self.fd = fd

    def __await__(self):
        yield self.op, self.fd


class Awaitable:
    def __await__(self):
        yield 1


async def make_task():
    print("pre_proc")
    await Awaitable()
    print("post proc")


if __name__ == '__main__':
    task = make_task()
    print(type(task))
    v = task.send(None)
    print(v)
    try:
        task.send(None)
    except StopIteration:
        print("StopIteration")
