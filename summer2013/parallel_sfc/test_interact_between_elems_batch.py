import subprocess


def test_interact_between_elems_avg():
    ne, ngq = 6, 4

    for nproc in xrange(1, ne*ne*6+1):
        cmd = 'mpirun -n %d python test_interact_between_elems.py %d %d' % (nproc, ne, ngq)
        ps = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = ps.communicate()
        print stdout
        assert stderr == '', '%s\n%s'%(cmd,stderr)
