#------------------------------------------------------------------------------
# filename  : scp_anaconda_pkgs.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.11.28    Start
#
#
# Description: 
#   SCP Anaconda pakcages
#   Anaconda environment and Paramiko should be installed.
#------------------------------------------------------------------------------

import paramiko
import os
from getpass import getpass


#
# Setup
#
host = 'gaon2'
id = 'khkim'
ip = {'gaon2':'172.128.66.201', \
      'gaon1':'172.128.66.223', \
      'bricks':'172.16.19.210'}[host]

src_basedir = '/home/{}/anaconda3/pkgs'.format(id)
dst_basedir = '/home/{}/usr/src/Anaconda3'.format(id)
pkg_list = [ \
        'netcdf4', \
        'basemap', \
        'pygrib', 'ecmwf_grib', 'jasper', 'libpng', 'pyproj', 'jpeg', \
        'pyspharm', \
        'mpi4py', 'mpich2', \
        'humanize']



#
# SSH
#
pw = getpass()

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

#kpath = os.path.expanduser('~/.ssh/id_rsa')
#mykey = paramiko.RSAKey.from_private_key_file(kpath)
#ssh.connect(ip, 22, id, pkey=mykey)
ssh.connect(ip, 22, id, pw)

stdin, stdout, stderr = ssh.exec_command('hostname')
print(stdout.read().decode('utf-8'))


#
# Transport
#
transport = paramiko.Transport((ip, 22))
transport.connect(username=id, password=pw)
sftp = paramiko.SFTPClient.from_transport(transport)

for pkg in pkg_list:
    stdin, stdout, stderr = ssh.exec_command('ls {}/{}-*.tar.bz2'.format(src_basedir, pkg))

    if stderr.read() != b'':
        print(stderr.read().decode('utf-8'))

    else:
        src_fpath_list = stdout.read().decode('utf-8').split('\n')

        for src_fpath in src_fpath_list[:-1]:
            print(src_fpath)
            fname = src_fpath.split('/')[-1]
            dst_fpath = os.path.join(dst_basedir, fname)

            exist = False
            if os.path.exists(dst_fpath):
                if os.stat(dst_fpath).st_size == sftp.stat(src_fpath).st_size:
                    exist = True

            if not exist:
                sftp.get(src_fpath, dst_fpath)

sftp.close()
