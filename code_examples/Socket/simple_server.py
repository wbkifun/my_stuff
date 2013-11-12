from socket import socket, gethostname

sk = socket()           # Create a socket object
host = gethostname()    # Get local machine name
port = 12345            # Reserve a port for your service
sk.bind((host, port))   # Bind to the port

sk.listen(5)            # Now wait for client connection (the maximum number of queued connections)
while True:
    conn, addr = sk.accept()    # Establish connection with client.
    print '[server] Got connection from', addr
    conn.send('Thank you for connecting')
    conn.close()                # Close the connection
