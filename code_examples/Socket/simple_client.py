from socket import socket, gethostname

sk = socket()           # Create a socket object
host = gethostname()    # Get local machine name
port = 12345            # Reserve a port for your service.

sk.connect((host, port))
print '[client] %s' % sk.recv(1024)
sk.close                # Close the socket when done
