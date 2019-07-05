import socket
import requests

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 10000)
print 'Starting up on %s port %s' % server_address
sock.bind(server_address)
sock.listen(1)

r = requests.post(url='192.168.86.186/tcp')
print r.json()

while True:
    # Wait for a connection
    print >>sys.stderr, 'waiting for a connection'
    connection, client_address = sock.accept()
    
    try:
        print >>sys.stderr, 'connection from', client_address

        # Receive the data in small chunks
        while True:
            data = connection.recv(16)
            print >>sys.stderr, 'received "%s"' % data
#            if data:
#                print >>sys.stderr, 'sending data back to the client'
#                connection.sendall(data)
#            else:
#                print >>sys.stderr, 'no more data from', client_address
#                break
            
    finally:
        # Clean up the connection
        connection.close()