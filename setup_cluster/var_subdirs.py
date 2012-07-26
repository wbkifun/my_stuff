#!/usr/bin/env python

import subprocess as sp
import os 


def ext_exec(cmd):
    # Execution the external command using subprocess module
    out, err = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.PIPE).communicate()
    if err != '': print err
    return out



def make_var_subdirs():
    paths = ext_exec('find /var')
    for path in paths.splitlines():
        if os.path.isdir(path):
            cmd = 'mkdir -m %s %s' % (oct(os.stat(path)[0])[2:], '/tmp' + path)		# ST_MODE
            ext_exec(cmd)

    ext_exec('mount -t tmpfs none /var')
    ext_exec('cp -a /tmp/var /')

    if os.path.isdir('/var/run/screen'):
        ext_exec('chmod 777 /var/run/screen')



if __name__ == '__main__':
    make_var_subdirs()
