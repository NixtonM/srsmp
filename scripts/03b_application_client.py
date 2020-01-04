import socket
import time
import struct
import sys

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    measurement_key = sys.argv[1] 
    s.sendall(bytes(str(measurement_key), 'utf8'))
    data = s.recv(1024)
    data_s = str(data,'utf8')
    return_code = int(data_s)
