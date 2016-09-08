import subprocess as subp
import os



def execute(cmd):
    print(cmd.split())
    ps = subp.Popen(cmd.split(), stdout=subp.PIPE, stderr=subp.PIPE)
    stdout, stderr = ps.communicate()
    sout = stdout.decode('utf-8')
    print(sout)
    #if 'Unrecognized' in sout: print(sout)
    assert len(stderr) == 0, "{}\n{}".format(sout, stderr.decode('utf-8'))



if __name__ == '__main__':
    cmd = 'ioc64 -device=cpu -cmd=compile -bo=-Ibuild/ -input=apb_ext.cl -ir=build/apb_ext.ir'
    execute(cmd)
    #os.system(cmd)

    cmd = "ioc64 -device='cpu' -cmd='compile' -bo='-Ibuild/' -input=apb.cl -ir=build/apb.ir"
    #execute(cmd)
    
    cmd = "ioc64 -device='cpu' -cmd='link' -binary='build/apb_ext.ir,build/apb.ir' -ir=build/apb.clbin"
    #execute(cmd)
