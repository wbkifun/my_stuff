from __future__ import division


ne, ngq = 3, 4

print 'idx\tgi\tgj\tei\tej\tface'


for idx in xrange(ngq*ngq*ne*ne*6):
    face = idx//(ngq*ngq*ne*ne) + 1
    ej = ( idx%(ngq*ngq*ne*ne) )//(ngq*ngq*ne) + 1
    ei = ( idx%(ngq*ngq*ne) )//(ngq*ngq) + 1
    gj = ( idx%(ngq*ngq) )//ngq + 1
    gi = idx%ngq + 1

    print idx, '\t\t', gi, gj, ei, ej, face
