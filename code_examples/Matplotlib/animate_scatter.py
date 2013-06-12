import numpy
import matplotlib.pyplot as plt


# data
nx, ny = 10, 10
x = numpy.random.rand(nx)
y = numpy.random.rand(ny)
z = 2*(x+y)


# plot
plt.ion()
fig = plt.figure(figsize=(12,8))
sc = plt.scatter(x, y, c=z, s=50)
plt.colorbar(sc)

for tstep in xrange(100):
    x += 0.002
    y += 0.001
    z = 2*(x+y)
    sc.set_offsets([x,y])               # set x, y
    sc.set_array(z)                     # set color
    #sc._sizes = numpy.ones(nx)*50       # set size
    plt.draw()

plt.show(True)
