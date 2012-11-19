import moddata

print 'mod.__doc__'
print moddata.mod.__doc__

moddata.mod.i = 5
moddata.mod.x[:2] = [1,2]
moddata.mod.a = [[1,2,3],[4,5,6]]

print 'exec foo()'
moddata.mod.foo()

print '\na='
print moddata.mod.a

print 'exec bar()'
moddata.mod.bar()

print 'set b'
moddata.mod.b = [[7,8,9], [18,9,2]]
moddata.mod.bar()

print 'set b with numpy'
import numpy as np
moddata.mod.b = np.random.random((3,4))
moddata.mod.bar()

print 'set b as None'
moddata.mod.b = None
moddata.mod.bar()
