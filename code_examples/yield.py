import numpy



def gen_sum_less_than(aa, bb, num):
    for i, a in enumerate(aa):
        for j, b in enumerate(bb):
            if a+b < num:
                yield i, j, a, b



# search table
arr1 = numpy.random.randint(1, 10, 5)   # low, high, size
arr2 = numpy.random.randint(1, 10, 5)

print 'arr1', arr1
print 'arr2', arr2


'''
for i, j, a, b in gen_sum_less_than(arr1, arr2, 5):
    print '(%d,%d), a=%d, b=%d, sum=%d' % (i,j,a,b,a+b)
'''

gen = gen_sum_less_than(arr1, arr2, 8)

while True:
    try:
        i,j,a,b = next(gen)
        print '(%d,%d), a=%d, b=%d, sum=%d' % (i,j,a,b,a+b)

    except StopIteration:
        break
