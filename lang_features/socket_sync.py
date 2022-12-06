import socket


def server(host, port):
    svr_fd = socket.socket()
    svr_fd.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    svr_fd.bind((host, port,))
    svr_fd.listen(5)

    while True:
        client, client_address = svr_fd.accept()
        client.send('@ connect\n'.encode())
        handler(client)


def handler(client):
    while True:
        req = client.recv(100)  # size of bytes chuck
        if not req:
            client.send('@ close connection\n'.encode())
            client.close()
            return

        try:
            value = int(req)
        except ValueError:
            client.send('@ enter integer\n'.encode())
            client.close()
            return

        resp = value * 2
        client.send(f'> {resp}\n'.encode())


if __name__ == '__main__':
    server('localhost', 30303)
