from __future__ import division
import numpy



rot90 = numpy.rot90
rot180 = lambda x: rot90(rot90(x))
rot270 = lambda x: rot90(rot90(rot90(x)))
inv_x = lambda x: x[::-1,:]     # inversion along the x axis
inv_y = lambda x: x[:,::-1]     # inversion along the y axis



#---------------------------------------------
# base curves
#---------------------------------------------
hilbert0 = numpy.array([ \
        [1,2], \
        [4,3]], 'i2', order='F')

peano0 = numpy.array([ \
        [1,4,5], \
        [2,3,6], \
        [9,8,7]], 'i2', order='F') 

cinco0 = numpy.array([ \
        [ 1, 8, 9,10,11], \
        [ 2, 7, 6,13,12], \
        [ 3, 4, 5,14,15], \
        [24,23,20,19,16], \
        [25,22,21,18,17]], 'i2', order='F') 



#---------------------------------------------
# derived curves
#---------------------------------------------
'''
Define the derived curves along the direction vectors

<direction vectors>
4 <--- 3
 ^    |
 |    v
1 ---> 2
'''

hilbert = {'name': 'hilbert', \
        '1->2': hilbert0, \
        '3->4': rot180(hilbert0), \
        '3->2': inv_y(rot90(hilbert0)), \
        '1->4': inv_y(rot270(hilbert0)) }

peano = {'name': 'peano', \
        '1->2': peano0, \
        '3->4': rot180(peano0), \
        '3->2': inv_y(rot90(peano0)), \
        '1->4': inv_y(rot270(peano0)) }

cinco = {'name': 'cinco', \
        '1->2': cinco0, \
        '3->4': rot180(cinco0), \
        '3->2': inv_y(rot90(cinco0)), \
        '1->4': inv_y(rot270(cinco0)) }



#---------------------------------------------
# direction vector table
#---------------------------------------------
'''
# cardinal points and direction vectors
    N
 4 <--- 3
W ^    | E
  |    v
 1 ---> 2
    S
'''

# previous direction : next direction
direction_vector_table = { \
         'E':{'W':'3->4', 'S':'3->2', 'N':'3->4'}, \
         'W':{'E':'1->2', 'S':'1->2', 'N':'1->4'}, \
         'S':{'E':'1->2', 'W':'1->4', 'N':'1->4'}, \
         'N':{'E':'3->2', 'W':'3->4', 'S':'3->2'} }





class SpaceFillingCurve(object):
    def __init__(self, N, ngq, nproc):
        assert ngq > 1

        self.N = N
        self.ngq = ngq
        self.nproc = nproc


        # factor list (2,3,5)
        self.factor_list = self.get_factor_list()


        # array variables
        self.sfc = numpy.zeros((N,N), 'i4', order='F')          # space-filling curve
        self.cube_gseq = numpy.zeros((N,N,6), 'i4', order='F')  # global sequence
        self.cube_proc = numpy.zeros((N,N,6), 'i4', order='F')  # process number
        self.nelems = numpy.zeros(nproc, 'i4')                  # number of elements


        # set the array variables
        self.set_sfc()              # space-filling curve on the square grid
        self.set_cube_gseq()        # global sequence on the cube
        self.set_cube_proc()        # partitioning the cube




    def get_factor_list(self):
        '''
        return factor list by the prime factors 2, 3, 5
        raise Error if the integer includes other prime factors
        '''

        N = self.N
        nproc = self.nproc

        prime_list = [2,3,5]
        if nproc%(6*2*2) == 0: prime_list = [3,5,2]
        elif nproc%(6*3*3) == 0: prime_list = [2,5,3]

        factor = list()
        for p in prime_list:
            while N%p == 0:
                factor.append(p)
                N //= p

        if N != 1:
            print_recommend_N(N)
            raise ValueError

        else:
            return factor




    def get_pos(self, curve, seq):
        '''
        get the position (i,j) of the sequence in the curve array
        '''

        ret = numpy.where(curve==seq)
        try:
            return [int(idx) for idx in ret]
        except TypeError:
            return [None for idx in ret]




    def get_direction_vector(self, lev, uplev_seq, prev_dirs, next_dirs, curves):
        get_pos = self.get_pos


        i0, j0 = get_pos(curves[lev+1], uplev_seq)
        i1, j1 = get_pos(curves[lev+1], uplev_seq+1)

        if i1 == None and j1 == None:
            next_dir = next_dirs[lev+1]
        else:
            next_dir = {(1,0):'E', (-1,0):'W', (0,1):'N', (0,-1):'S'}[i1-i0,j1-j0]

        prev_dir = {'E':'W', 'W':'E', 'S':'N', 'N':'S'}[next_dirs[lev]]
        prev_dirs[lev] = prev_dir
        next_dirs[lev] = next_dir
        
        direction_vector = direction_vector_table[prev_dir][next_dir]

        return direction_vector




    def set_sfc(self):
        N = self.N
        facts = self.factor_list
        sfc = self.sfc
        get_direction_vector = self.get_direction_vector


        # initialize
        dv_curves = [{2:hilbert, 3:peano, 5:cinco}[fact] for fact in facts]
        curves = [dv_curve['1->2'] for dv_curve in dv_curves]
        prev_dirs = ['W' for fact in facts]
        next_dirs = ['E' for fact in facts]
        sizes = [curve.size for curve in curves]
        periods = numpy.multiply.accumulate([1] + [size for size in sizes[1:]])
        nlev = len(facts)
        dx = facts[0]
        gi, gj= -dx, 0

        # generate curves
        for seq in xrange( sfc.size//sizes[0] ):
            for lev in xrange(nlev-2,-1,-1):
                if seq % periods[lev] == 0:
                    uplev_seq = (seq // periods[lev]) % sizes[lev+1] + 1
                    mv = get_direction_vector( \
                            lev, uplev_seq, prev_dirs, next_dirs, curves)
                    curves[lev] = dv_curves[lev][mv]
            
            gi, gj= {'E':(gi-dx,gj), 'W':(gi+dx,gj), \
                     'S':(gi,gj+dx), 'N':(gi,gj-dx)}[prev_dirs[0]]
            sfc[gi:gi+dx,gj:gj+dx] = curves[0] + seq*sizes[0]




    def set_cube_gseq(self):
        '''
        # direction vectors on the cube
                  ----
                 | 3  |
                 |    |
                 1<---2
        
          ----3   ----   4----    ----3
         | 4  |  | 1  |  ^ 2  |  | 6  ^
         |    V  |    |  |    |  |    |
          ----2  1--->2  1----    ----2
        
                 4----
                 | 5  |
                 V    |
                 1----
        '''

        N = self.N
        cube_gseq = self.cube_gseq

        sfc1 = self.sfc
        sfc2 = N**2   + inv_y( rot270(sfc1) )
        sfc3 = 2*N**2 + inv_x( sfc1 )
        sfc4 = 3*N**2 + inv_y( rot90(sfc1) )
        sfc5 = 4*N**2 + rot270( sfc1 )
        sfc6 = 5*N**2 + rot90( sfc1 )

        for i in xrange(6):
            cube_gseq[:,:,i] = locals()['sfc%d'%(i+1)]




    def set_cube_proc(self):
        '''
        Numbering the process number
        '''

        N = self.N
        nproc = self.nproc
        cube_gseq = self.cube_gseq
        cube_proc = self.cube_proc
        nelems = self.nelems

        gs = N*N*6  # global size
        nelems[:] = numpy.array( [gs//nproc for i in xrange(nproc)] )
        for i in xrange( gs%nproc ): nelems[i] += 1

        accum = numpy.add.accumulate([1] + list(nelems))
        for proc in xrange(nproc):
            condition = (cube_gseq>=accum[proc]) * (cube_gseq<accum[proc+1])
            cube_proc[condition] = proc + 1





def print_recommend_N(N):
    import os
    import sys


    Nmax = 3500     # N=3340 if dx ~= 1 km

    file_path = globals()['__file__']
    base_path = file_path[:file_path.rfind('/')+1]
    fpath = base_path + 'avail_N_list_%d.pkl' % Nmax
    if os.path.exists(fpath):
        avail_Ns = pickle.load( open(fpath, 'r') )
    else:
        avail_Ns = dict()
        for N in xrange(1,Nmax):
            facts = factor235(N)
            
            if facts != None:
                avail_Ns[N] = facts

        fpath = base_path + 'avail_N_list_%d.pkl' % Nmax
        f = open(fpath, 'w')
        pickle.dump(avail_Ns, f)
        f.close()


    for i in xrange(N, 0 ,-1):
        if avail_Ns.has_key(i):
            N1 = i
            N1_facts = avail_Ns.get(i)
            break

    for i in xrange(N, Nmax):
        if avail_Ns.has_key(i):
            N2 = i
            N2_facts = avail_Ns.get(i)
            break

    print('Error: %d cannot be factorized by 5, 3, 2.' % N)
    print('We recommend the nearest avaiable N as bellow:')
    print('%d  %s' % (N1, N1_facts))
    print('%d  %s' % (N2, N2_facts))

    sys.exit()




if __name__ == '__main__':
    N, npq = 3, 4
    nproc = 3
    sfc = SpaceFillingCurve(N, npq, nproc)
    print sfc.cube_proc[:,:,0]
