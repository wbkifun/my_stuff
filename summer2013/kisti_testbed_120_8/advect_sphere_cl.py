from __future__ import division
import numpy
import pyopencl as cl
import sys
from numpy import pi, sin, cos, arccos, exp, abs
from numpy import int32 as i4, float64 as f8



#------------------------------------------------------------------------------
# Platform, Device, Context and Queue
#------------------------------------------------------------------------------
platforms = cl.get_platforms()

# MIC machine (KISTI)
'''
intel = platforms[0]
devices = intel.get_devices()

for dev in devices: print dev
cpu, mic = devices
#context = cl.Context(devices)
context_cpu = cl.Context([cpu])
context_mic = cl.Context([mic])
queue_cpu = cl.CommandQueue(context_cpu, cpu)
queue_mic = cl.CommandQueue(context_mic, mic)
 
device = cpu
context = context_cpu
queue = queue_cpu
'''

# GPU machine (KIAPS)
intel, nvidia = platforms
intel_devices = intel.get_devices()
nvidia_devices = nvidia.get_devices()

for dev in intel_devices: print dev
for dev in nvidia_devices: print dev
cpu = intel_devices[0]
gpu = nvidia_devices[0]
#context = cl.Context(devices)
context_cpu = cl.Context(intel_devices)
context_gpu = cl.Context(nvidia_devices)
queue_cpu = cl.CommandQueue(context_cpu, cpu)
queue_gpu = cl.CommandQueue(context_gpu, gpu)
 
device = cpu
context = context_cpu
queue = queue_cpu


mf_rw = cl.mem_flags.READ_WRITE
mf_ro = cl.mem_flags.READ_ONLY


#------------------------------------------------------------------------------
# program and kernel
#------------------------------------------------------------------------------
kernels = open('./sem_cores.cl').read()
prg = cl.Program(context, kernels).build(options='')

#preferred_multiple = cl.Kernel(prg, 'force').get_work_group_info( \
#	cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device) 




class State(object):
    def __init__(self, ngll, nelem):
        ngll = ngll
        nelem = nelem

        self.psi = psi = numpy.zeros((ngll,ngll,1,nelem), 'f8', order='F')
        velocity = numpy.zeros((2,ngll,ngll,1,nelem), 'f8', order='F')
        psi[:] = numpy.load('./preprocess_files/psi.npy')
        velocity[:] = numpy.load('./preprocess_files/velocity.npy')

        # CL buffers
        self.psi_buf = cl.Buffer(context, mf_rw, psi.nbytes)
        self.velocity_buf = cl.Buffer(context, mf_rw, velocity.nbytes)

        cl.enqueue_copy(queue, self.psi_buf, psi)
        cl.enqueue_copy(queue, self.velocity_buf, velocity)




class InteractBetweenElems(object):
    def __init__(self, ngll):
        self.ngll = ngll

        # fixed for N=120, ngll=4
        mvp_inner2_tmp = numpy.load('./preprocess_files/mvp_inner2.npy')
        mvp_inner3_tmp = numpy.load('./preprocess_files/mvp_inner3.npy')
        mvp_inner4_tmp = numpy.load('./preprocess_files/mvp_inner4.npy')

        mvp_inner2 = numpy.zeros(mvp_inner2_tmp.shape, 'i4', order='F')
        mvp_inner3 = numpy.zeros(mvp_inner3_tmp.shape, 'i4', order='F')
        mvp_inner4 = numpy.zeros(mvp_inner4_tmp.shape, 'i4', order='F')

        mvp_inner2[:] = mvp_inner2_tmp[:]
        mvp_inner3[:] = mvp_inner3_tmp[:]
        mvp_inner4[:] = mvp_inner4_tmp[:]


        # CL buffers
        self.mvp_inner2_buf = cl.Buffer(context, mf_ro, mvp_inner2.nbytes)
        self.mvp_inner3_buf = cl.Buffer(context, mf_ro, mvp_inner3.nbytes)
        self.mvp_inner4_buf = cl.Buffer(context, mf_ro, mvp_inner4.nbytes)

        cl.enqueue_copy(queue, self.mvp_inner2_buf, mvp_inner2)
        cl.enqueue_copy(queue, self.mvp_inner3_buf, mvp_inner3)
        cl.enqueue_copy(queue, self.mvp_inner4_buf, mvp_inner4)

        self.gs2 = mvp_inner2.shape[-1]
        self.gs3 = mvp_inner3.shape[-1]
        self.gs4 = mvp_inner4.shape[-1]



    def interact_between_elems_inner(self, var_buf):
        mvp_inner2_buf = self.mvp_inner2_buf
        mvp_inner3_buf = self.mvp_inner3_buf
        mvp_inner4_buf = self.mvp_inner4_buf
        gs2, gs3, gs4 = self.gs2, self.gs3, self.gs4
        ngll = self.ngll

        prg.interact_inner2(queue, (gs2,), None, i4(gs2), i4(ngll), mvp_inner2_buf, var_buf)
        prg.interact_inner3(queue, (gs3,), None, i4(gs3), i4(ngll), mvp_inner3_buf, var_buf)
        prg.interact_inner4(queue, (gs4,), None, i4(gs4), i4(ngll), mvp_inner4_buf, var_buf)




class InsideElems(object):
    def __init__(self, N, ngll, nelem, state, interact):
        self.N = N 
        self.ngll = ngll 
        self.nelem = nelem 
        self.state = state
        self.interact = interact


        # transform matrix
        AI = numpy.zeros((2,2,ngll,ngll,nelem), 'f8', order='F')
        J = numpy.zeros((ngll,ngll,nelem), 'f8', order='F')
        AI[:] = numpy.load('./preprocess_files/AI.npy')
        J[:] = numpy.load('./preprocess_files/J.npy')


        # derivative matrix
        dvvT = numpy.zeros((ngll,ngll), 'f8', order='F')
        dvvT[:] = numpy.load('./preprocess_files/dvvT.npy')


        # CL buffers
        self.AI_buf = cl.Buffer(context, mf_ro, AI.nbytes)
        self.J_buf = cl.Buffer(context, mf_ro, J.nbytes)
        self.dvvT_buf = cl.Buffer(context, mf_ro, dvvT.nbytes)

        cl.enqueue_copy(queue, self.AI_buf, AI)
        cl.enqueue_copy(queue, self.J_buf, J)
        cl.enqueue_copy(queue, self.dvvT_buf, dvvT)

        self.gs = ngll*ngll*nelem



    def compute_rhs(self, psi_buf, ret_psi_buf):
        nelem = self.nelem
        ngll = self.ngll
        dvvT_buf = self.dvvT_buf
        J_buf = self.J_buf
        AI_buf = self.AI_buf
        velocity_buf = self.state.velocity_buf
        gs = self.gs

        prg.compute_rhs(queue, (gs,), None, i4(nelem), i4(ngll), dvvT_buf, J_buf, AI_buf, velocity_buf, psi_buf, ret_psi_buf)
        self.interact.interact_between_elems_inner(ret_psi_buf)




class RungeKutta(object):
    def __init__(self, dt, psi, inside):
        self.dt = dt
        self.tn = dt

        self.k1_buf = cl.Buffer(context, mf_rw, psi.nbytes)
        self.k2_buf = cl.Buffer(context, mf_rw, psi.nbytes)
        self.k3_buf = cl.Buffer(context, mf_rw, psi.nbytes)
        self.k4_buf = cl.Buffer(context, mf_rw, psi.nbytes)
        self.psi2_buf = cl.Buffer(context, mf_rw, psi.nbytes)

        self.inside = inside



    def update_rk4(self, psi_buf):
        dt = self.dt
        tn = self.tn
        k1, k2 = self.k1_buf, self.k2_buf
        k3, k4 = self.k3_buf, self.k4_buf
        psi2 = self.psi2_buf

        psi = psi_buf
        gs = self.inside.gs


        self.inside.compute_rhs(psi, k1) 
  
        prg.add(queue, (gs,), None, i4(gs), f8(0.5*dt), k1, psi, psi2)
        self.inside.compute_rhs(psi2, k2) 

        prg.add(queue, (gs,), None, i4(gs), f8(0.5*dt), k2, psi, psi2)
        self.inside.compute_rhs(psi2, k3) 

        prg.add(queue, (gs,), None, i4(gs), f8(dt), k3, psi, psi2)
        self.inside.compute_rhs(psi2, k4) 

        prg.rk4_add(queue, (gs,), None, i4(gs), f8(dt), k1, k2, k3, k4, psi)

        tn += dt




if __name__ == '__main__':
    #----------------------------------------------
    # setup
    #----------------------------------------------
    N = 120         # elements / axis
    ngll = 8        # GLL points / axis / element
    cfl = 0.1       # Courant-Friedrichs-Lewy condition 
    nproc = 1
    rank = 1

    nelem = N*N*6
    #min_dx = 0.0034069557991275559  # fixed for N=120, ngll=4
    min_dx = 0.00079122793323062573  # fixed for N=120, ngll=4

    # state variables
    state = State(ngll, nelem)


    #----------------------------------------------
    # spectral element
    #----------------------------------------------
    interact = InteractBetweenElems(ngll)
    inside = InsideElems(N, ngll, nelem, state, interact)

    # minimum dx, dt
    max_v = 2*pi
    dt = cfl*min_dx/max_v
    tloop = RungeKutta(dt, state.psi, inside)


    #----------------------------------------------
    # print the setup information
    #----------------------------------------------
    print '-'*47
    print 'N\t\t', N
    print 'ngll\t\t', ngll
    print 'nelem\t\t', nelem
    print 'cfl\t\t', cfl
    print 'min_dx\t\t', min_dx
    print 'dt\t\t', dt

    #tmax = int( numpy.ceil(1/dt) )
    tmax = 1000
    tgap = 100

    print 'tmax\t\t', tmax
    print 'tgap\t\t', tgap
    print '-'*47
    print ''

    """
    ret_psi = inside.compute_rhs(0, state.psi)
    numpy.save('compute_rhs_psi0.npy', state.psi)
    numpy.save('compute_rhs_psi1.npy', ret_psi)

    numpy.save('./run/%.6d_rank%d_psi.npy' % (0, rank), state.psi)
    numpy.save('./run/%.6d_rank%d_velocity.npy' % (0, rank), state.velocity)
    """

    #----------------------------------------------
    # time loop
    #----------------------------------------------
    for tstep in xrange(1,tmax+1):
        tloop.update_rk4(state.psi_buf)

        if tstep%tgap == 0:
            print 'tstep=\t%d/%d (%g %s)\r' % (tstep, tmax, tstep/tmax*100, '%'),
            sys.stdout.flush()

            #numpy.save('./run/%.6d_rank%d_psi.npy' % (tstep, rank), state.psi)

	cl.enqueue_copy(queue, state.psi, state.psi_buf)
    print ''
