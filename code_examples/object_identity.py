class AA(object):
    def __init__(self, v):
        self.v = v

    def set_v(self, v):
        self.v = v



a1 = AA(1)
a2 = AA(2)
a3 = AA(3)
a4 = AA(4)

print ''
print 'a1 == a1', a1 == a1
print 'a1 is a1', a1 is a1
print ''
print 'a1 == a2', a1 == a2
print 'a1 is a2', a1 is a2

a1.set_v('abcd')
print ''
print 'a1.set_v(..)'
print 'a1 == a1', a1 == a1
print 'a1 is a1', a1 is a1

aas = [a1, a2, a3, a4]
print ''
print 'a1 in [a1,a2,a3,a4]', a1 in aas

ss = set([a1,a2,a3,a4,a3,a4,a1])
print ''
print 'len( set([a1,a2,a3,a4,a3,a4,a1]) ) =', len(ss)


dd = {a1:1, a2:2, a3:3}
print ''
print 'Define dictionary'
print 'dd = {a1:1, a2:2, a3:3}'
print 'dd[a2]=', dd[a2]
