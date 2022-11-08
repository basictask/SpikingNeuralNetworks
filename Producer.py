import numpy as np
import socket
import time


class Producer:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("localhost", 34567))
        self.sock.listen(5)
        (self.conn, address) = self.sock.accept()

    def send(self, dataMat):
        self.conn.send(dataMat)


if __name__ == "__main__":
    prod = Producer()
