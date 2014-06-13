import numpy


# (3,4,size)
mvp_inner = numpy.load('./preprocess_files/mvp_inner.npy')
size = mvp_inner.shape[-1]


n2, n3, n4 = 0, 0, 0

for k in xrange(size):
    if mvp_inner[0,2,k] == 0: 
        n2 += 1
    elif mvp_inner[0,3,k] == 0:
        n3 += 1
    else:
        n4 += 1



mvp_inner2 = numpy.zeros((3,2,n2), 'i4', order='F')
mvp_inner3 = numpy.zeros((3,3,n3), 'i4', order='F')
mvp_inner4 = numpy.zeros((3,4,n4), 'i4', order='F')


k2, k3, k4 = 0, 0, 0

for k in xrange(size):
    if mvp_inner[0,2,k] == 0: 
        mvp_inner2[:,:,k2] = mvp_inner[:,:2,k]  
        k2 += 1

    elif mvp_inner[0,3,k] == 0:
        mvp_inner3[:,:,k3] = mvp_inner[:,:3,k]
        k3 += 1

    else:
        mvp_inner4[:,:,k4] = mvp_inner[:,:,k]
        k4 += 1


numpy.save('./preprocess_files/mvp_inner2.npy', mvp_inner2)
numpy.save('./preprocess_files/mvp_inner3.npy', mvp_inner3)
numpy.save('./preprocess_files/mvp_inner4.npy', mvp_inner4)
