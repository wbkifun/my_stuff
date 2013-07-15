#------------------------------------------------------------------------------
# Lennard-Jones potential
# Periodic boundary condition
#------------------------------------------------------------------------------

from __future__ import division
import numpy
import pickle
from numpy import sqrt



class LennardJones(object):
    def __init__(self, nn):
        self.nn = nn    # number of atoms, molecules


        self.domain = 300    # domain size (a.u.)
        self.dt = 10         # time interval (a.u.)
        self.ms = 0.0        # max speed (a.u.)
        self.em = 1822.88839*28.0134     # effective mass of N2
        self.lje = 0.000313202           # Lennard-Jones epsilon of N2
        self.ljs = 6.908841465           # Lennard-Jones sigma of N2


        self.allocate_initialize()


        
    def allocate_initialize(self):
        nn = self.nn
        domain = self.domain
        ms = self.ms


        fname = 'x_v_nn%d.pkl' % nn
        try:
            x, y, z, vx, vy, vz = pickle.load( open(fname) )

        except:
            x = domain * numpy.random.rand(nn)
            y = domain * numpy.random.rand(nn)
            z = domain * numpy.random.rand(nn)
            vx = ms * (numpy.random.rand(nn) - 0.5)
            vy = ms * (numpy.random.rand(nn) - 0.5)
            vz = ms * (numpy.random.rand(nn) - 0.5)

            f = open(fname, 'wb')
            pickle.dump([x,y,z,vx,vy,vz], f)
            f.close()

        fx = numpy.zeros(nn)
        fy = numpy.zeros(nn)
        fz = numpy.zeros(nn)


        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.fx, self.fy, self.fz = fx, fy, fz



    def force(self):
        nn = self.nn
        lje, ljs = self.lje, self.ljs
        x, y, z = self.x, self.y, self.z
        vx, vy, vz = self.vx, self.vy, self.vz
        fx, fy, fz = self.fx, self.fy, self.fz

        coe = -24 * lje
        

        for i in xrange(nn):
            fx[i] = 0
            fy[i] = 0
            fz[i] = 0

            for j in xrange(nn):
                if i == j: continue

                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]

                r = sqrt(dx*dx + dy*dy* + dz*dz)
                r2 = r*r

                pe = 2*pow(ljs/r,12) - pow(ljs/r,6)
                fx[i] += pe*dx/r2
                fy[i] += pe*dy/r2
                fz[i] += pe*dz/r2

            fx[i] *= coe
            fy[i] *= coe
            fz[i] *= coe



    def solve(self):
        '''
        Euler solver
        '''

        nn = self.nn
        dt = self.dt
        em = self.em
        x, y, z = self.x, self.y, self.z
        vx, vy, vz = self.vx, self.vy, self.vz
        fx, fy, fz = self.fx, self.fy, self.fz


        for i in xrange(nn):
            x[i] += vx[i]*dt
            y[i] += vy[i]*dt
            z[i] += vz[i]*dt

            vx[i] += fx[i]*dt / em
            vy[i] += fy[i]*dt / em
            vz[i] += fz[i]*dt / em



    def output_energy(self):
        nn = self.nn
        em = self.em
        lje, ljs = self.lje, self.ljs
        x, y, z = self.x, self.y, self.z
        vx, vy, vz = self.vx, self.vy, self.vz


        # kinetic energy
        ke = 0
        for i in xrange(nn):
            ke += vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        ke *= 0.5*em

        # potential energy
        pe = 0
        for i in xrange(nn):
            for j in xrange(nn):
                if i == j: continue
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]

                r = sqrt(dx*dx + dy*dy* + dz*dz)
                pe += 2 * lje * (pow(ljs/r,12) - pow(ljs/r,6))

        te = ke + pe

        return ke, pe, te




if __name__ == '__main__':
    nn = 500        # number of atoms, molecules
    tmax = 20       # max time step

    LJ = LennardJones(nn)

    print '#\tKinEng\tPotEng\tTotEng'
    for tstep in xrange(1,tmax+1):
        LJ.force()
        LJ.solve()

        ke, pe, te = LJ.output_energy()
        print '%d\t%1.5f\t%1.5f\t%1.5f' % (tstep, ke, pe, te)

        '''
        xyz_list = [LJ.x, LJ.y, LJ.z]
        f = open('xyz_nn%d_tstep%d.pkl', 'wb')
        pickle.dump(xyz_list, f)
        f.close()
        '''
