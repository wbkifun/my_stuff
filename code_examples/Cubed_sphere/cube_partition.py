#------------------------------------------------------------------------------
# filename  : cube_partition.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2013.9.8      revision
#
#
# description: 
#   Generate the partitions for the cubed-sphere using Space Filling Curves
#
# class:
#   CubePartition()
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np



rot90 = np.rot90
rot180 = lambda x: rot90(rot90(x))
rot270 = lambda x: rot90(rot90(rot90(x)))
inv_x = lambda x: x[::-1,:]     # inversion along the x axis
inv_y = lambda x: x[:,::-1]     # inversion along the y axis




class CubePartition(object):
    def __init__(self, ne, nproc):
        self.ne = ne
        self.nproc = nproc

        self.init_base_curves()
        self.init_derived_curves()
        self.init_direction_vector_table()


        # factor list (2,3,5)
        self.factor_list = self.get_factor_list()


        # array variables
        self.sfc = np.zeros((ne,ne), 'i4')          # space-filling curve
        self.elem_gseq = np.zeros((6,ne,ne), 'i4')  # global sequence
        self.elem_proc = np.zeros((6,ne,ne), 'i4')  # process number
        self.nelems = np.zeros(nproc, 'i4')         # number of elements

        # set the array variables
        self.set_sfc()              # space-filling curve on the square grid
        self.set_elem_gseq()        # global sequence on the cube
        self.set_elem_proc()        # partitioning the cube
        


    def init_base_curves(self):
        self.hilbert0 = np.array([ \
                [1,2], \
                [4,3]], 'i2')

        self.peano0 = np.array([ \
                [1,4,5], \
                [2,3,6], \
                [9,8,7]], 'i2') 

        self.cinco0 = np.array([ \
                [ 1, 8, 9,10,11], \
                [ 2, 7, 6,13,12], \
                [ 3, 4, 5,14,15], \
                [24,23,20,19,16], \
                [25,22,21,18,17]], 'i2') 



    def init_derived_curves(self):
        '''
        Define the derived curves along the direction vectors

        <edge coordinates>
        4 ---- 3
         |    |
         |    |
        1 ---- 2
        '''

        hilbert0 = self.hilbert0
        peano0 = self.peano0
        cinco0 = self.cinco0

        self.hilbert = {'name': 'hilbert', \
                '1->2': hilbert0, \
                '3->4': rot180(hilbert0), \
                '3->2': inv_y(rot90(hilbert0)), \
                '1->4': inv_y(rot270(hilbert0)) }

        self.peano = {'name': 'peano', \
                '1->2': peano0, \
                '3->4': rot180(peano0), \
                '3->2': inv_y(rot90(peano0)), \
                '1->4': inv_y(rot270(peano0)) }

        self.cinco = {'name': 'cinco', \
                '1->2': cinco0, \
                '3->4': rot180(cinco0), \
                '3->2': inv_y(rot90(cinco0)), \
                '1->4': inv_y(rot270(cinco0)) }



    def init_direction_vector_table(self):
        '''
        Define the table for the direction vector

        # direction coordinates
             N
          4 ---- 3
        W  |    |  E
           |    |
          1 ---- 2
             S

        # relation diagram with the previous element

                  4 ---- 3
                   |    |
                   |    |
                  1 ---- 2

        4 ---- 3  4 ---- 3  4 ---- 3
         |    |    |prev|    |    |
         |    |    |    |    |    |
        1 ---- 2  1 ---- 2  1 ---- 2

                  4 ---- 3
                   |    |
                   |    |
                  1 ---- 2
        '''

        # previous direction : next direction
        self.direction_vector_table = \
                {'E':{'W':'3->4', 'S':'3->2', 'N':'3->4'}, \
                 'W':{'E':'1->2', 'S':'1->2', 'N':'1->4'}, \
                 'S':{'E':'1->2', 'W':'1->4', 'N':'1->4'}, \
                 'N':{'E':'3->2', 'W':'3->4', 'S':'3->2'} }



    def get_factor_list(self):
        '''
        return factor list by the prime factors 2, 3, 5
        return None if the integer includes other prime factors
        '''

        ne = self.ne
        nproc = self.nproc

        prime_list = [2,3,5]
        if nproc%(6*2*2) == 0: prime_list = [3,5,2]
        elif nproc%(6*3*3) == 0: prime_list = [2,5,3]

        factor = list()
        for p in prime_list:
            while ne%p == 0:
                factor.append(p)
                ne //= p

        assert ne==1, 'Error: The ne must have the prime factors 2, 3 and 5.'
        return factor



    def get_pos(self, curve, seq):
        '''
        get the position (i,j) of the sequence in the curve array
        '''

        ret = np.where(curve==seq)
        try:
            return [int(idx) for idx in ret]
        except TypeError:
            return [None for idx in ret]



    def get_direction_vector(self, lev, uplev_seq, prev_dirs, next_dirs, curves):
        i0, j0 = self.get_pos(curves[lev+1], uplev_seq)
        i1, j1 = self.get_pos(curves[lev+1], uplev_seq+1)

        if i1 == None and j1 ==None:
            next_dir = next_dirs[lev+1]
        else:
            next_dir = {(1,0):'E', (-1,0):'W', (0,1):'N', (0,-1):'S'}[i1-i0,j1-j0]

        prev_dir = {'E':'W', 'W':'E', 'S':'N', 'N':'S'}[next_dirs[lev]]
        prev_dirs[lev] = prev_dir
        next_dirs[lev] = next_dir

        return self.direction_vector_table[prev_dir][next_dir]



    def set_sfc(self):
        ne = self.ne
        facts = self.factor_list
        sfc = self.sfc

        # initialize
        dv_curves = [{2:self.hilbert, 3:self.peano, 5:self.cinco}[fact] \
                for fact in facts]
        curves = [dv_curve['1->2'] for dv_curve in dv_curves]
        prev_dirs = ['W' for fact in facts]
        next_dirs = ['E' for fact in facts]
        sizes = [curve.size for curve in curves]
        periods = np.multiply.accumulate([1] + [size for size in sizes[1:]])
        nlev = len(facts)
        dx = facts[0]
        gi, gj= -dx, 0

        # generate curves
        for seq in xrange( sfc.size//sizes[0] ):
            for lev in xrange(nlev-2,-1,-1):
                if seq % periods[lev] == 0:
                    uplev_seq = (seq // periods[lev]) % sizes[lev+1] + 1
                    mv = self.get_direction_vector( \
                            lev, uplev_seq, prev_dirs, next_dirs, curves)
                    curves[lev] = dv_curves[lev][mv]
            
            gi, gj= {'E':(gi-dx,gj), 'W':(gi+dx,gj), \
                     'S':(gi,gj+dx), 'N':(gi,gj-dx)}[prev_dirs[0]]
            sfc[gi:gi+dx,gj:gj+dx] = curves[0] + seq*sizes[0]



    def set_elem_gseq(self):
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

        ne = self.ne
        elem_gseq = self.elem_gseq

        sfc1 = self.sfc
        sfc2 = ne**2   + inv_y( rot270(sfc1) )
        sfc3 = 2*ne**2 + inv_x( sfc1 )
        sfc4 = 3*ne**2 + inv_y( rot90(sfc1) )
        sfc5 = 4*ne**2 + rot270( sfc1 )
        sfc6 = 5*ne**2 + rot90( sfc1 )

        for i in xrange(6):
            elem_gseq[i,:,:] = locals()['sfc%d'%(i+1)]



    def set_elem_proc(self):
        '''
        Numbering the process number
        '''

        ne = self.ne
        nproc = self.nproc
        elem_gseq = self.elem_gseq
        elem_proc = self.elem_proc
        nelems = self.nelems

        gs = ne*ne*6  # global size
        nelems[:] = np.array( [gs//nproc for i in xrange(nproc)] )
        for i in xrange( gs%nproc ): nelems[i] += 1

        accum = np.add.accumulate([1] + list(nelems))
        for proc in xrange(nproc):
            condition = (elem_gseq>=accum[proc]) * (elem_gseq<accum[proc+1])
            elem_proc[condition] = proc
