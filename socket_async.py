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


class TaskManager:
    def __init__(self):
        self.tasks = []
        self.reads = {}
        self.writes = {}

    def add_task(self, task):
        self.tasks.append(task)

    def select(self):
        rs, ws, _ = select.select(self.reads, self.writes, [])
        for r in rs:
            self.add_task(self.reads.pop(r))
        for w in ws:
            self.add_task(self.writes.pop(w))

    def handle_operation(self, current_task, op, fd):
        if op is OpTypes.READ_OP:
            self.reads[fd] = current_task
        elif op is OpTypes.WRITE_OP:
            self.writes[fd] = current_task
        else:
            pass

    def run(self):
        while any((self.tasks, self.reads, self.writes)):
            while not self.tasks:
                self.select()
            current_task = self.tasks.pop(0)
            try:
                op, fd = current_task.send(None)
            except StopIteration:
                continue
            self.handle_operation(current_task, op, fd)


class AsyncSocket:
    def __init__(self, fd):
        self.fd = fd

    async def accept(self):
        await AwaitOp(OpTypes.READ_OP, self.fd)
        return self.fd.accept()

    async def recv(self, length):
        await AwaitOp(OpTypes.READ_OP, self.fd)
        return self.fd.recv(length)

    async def send(self, data):
        if isinstance(data, str):
            data = data.encode()
        await AwaitOp(OpTypes.WRITE_OP, self.fd)
        return self.fd.send(data)

    def close(self):
        self.fd.close()


async def server(task_mgr, host, port):
    svr_fd = socket.socket()
    svr_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    svr_fd.bind((host, port,))
    svr_fd.listen(5)
    svr_fd = AsyncSocket(svr_fd)
    while True:
        client, client_address = await svr_fd.accept()
        client = AsyncSocket(client)
        await client.send('@ connect\n'.encode())
        task_mgr.add_task(handler(client))


async def handler(client):
    while True:
        req = await client.recv(100)  # size of bytes chuck
        if not req:
            await client.send('@ close connection\n'.encode())
            client.close()
            return

        try:
            value = int(req)
        except ValueError:
            await client.send('@ enter integer\n'.encode())
            client.close()
            return

        resp = value * 2
        await client.send(f'> {resp}\n'.encode())


if __name__ == '__main__':
    task_mgr = TaskManager()
    task_mgr.add_task(server(task_mgr, 'localhost', 30303))
    task_mgr.run()


class Awaitable:
    def __await__(self):
        yield 1


async def make_task():
    print("pre_proc")
    await Awaitable()
    print("post proc")


def task_send_test():
    task = make_task()
    print(type(task))
    v = task.send(None)
    print(v)
    try:
        task.send(None)
    except StopIteration:
        print("StopIteration")
