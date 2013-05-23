import paramiko

gaon1 = '118.128.66.133'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(gaon1, username='khkim', password='fje3*kf')
stdin, stdout, stderr = ssh.exec_command('uptime')

print 'stdin', stdin
print 'stdout', stdout.readlines()
print 'stderr', stderr.readlines()
