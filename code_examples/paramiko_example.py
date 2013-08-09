import paramiko

server = '192.168.10.1'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(server, username='myid', password='myidpasswd')
stdin, stdout, stderr = ssh.exec_command('uptime')

print 'stdin', stdin
print 'stdout', stdout.readlines()
print 'stderr', stderr.readlines()
