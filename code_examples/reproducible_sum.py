from __future__ import division
import numpy
from numpy.testing import assert_array_equal as assert_ae
from numpy.testing import assert_array_almost_equal as assert_aae

numpy.set_printoptions(precision=16)



def naive_sum(arr1, arr2):
    sum1 = arr1.sum()
    sum2 = arr2.sum()

    return sum1, sum2



def convert_quad_sum(arr1, arr2):
    qsum1 = numpy.float128(0)
    qsum2 = numpy.float128(0)
    for i in xrange(nx):
        qsum1 += numpy.float128(arr1[i])
        qsum2 += numpy.float128(arr2[i])

    sum1 = numpy.float64(qsum1)
    sum2 = numpy.float64(qsum2)

    return sum1, sum2



def convert_quad_sum_pyramid(arr1, arr2, power):
    qarr1 = numpy.float128(arr1)
    qarr2 = numpy.float128(arr2)
    for p in [2**i for i in xrange(1,power+1)]:
        for i in xrange(nx//p):
            qarr1[p*i] += qarr1[p*i+p//2]
            qarr2[p*i] += qarr2[p*i+p//2]

    sum1 = numpy.float64(qarr1[0])
    sum2 = numpy.float64(qarr2[0])

    return sum1, sum2



def check_equal(sum1, sum2):
    try: assert_aae(sum1, sum2, 15)
    except: print '\tmismatch'
    else: print '\tok'




if __name__=='__main__':
    # setup
    power = 20
    nx = 2**power
    print 'nx=%d' % nx

    for i in xrange(100):
        print 'i=%d' % i

        arr1 = numpy.random.uniform(0, 100000000, nx)
        arr2 = arr1.copy()
        numpy.random.shuffle(arr2)


        print 'naive_sum()', 
        sum1, sum2 = naive_sum(arr1, arr2)
        check_equal(sum1, sum2)


        print 'convert_quad_sum()', 
        sum1, sum2 = convert_quad_sum(arr1, arr2)
        check_equal(sum1, sum2)


        print 'convert_quad_sum_pyramid()', 
        sum1, sum2 = convert_quad_sum_pyramid(arr1, arr2, power)
        check_equal(sum1, sum2)
