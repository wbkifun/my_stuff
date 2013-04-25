from __future__ import division
import numpy
import matplotlib.pyplot as plt


from smooth import smoothing as smoothing_f90


def noisy_data(n):
    T = 40  # time interval (0,T)
    dt = T/n
    t = numpy.linspace(0,T,n+1)
    y = numpy.sin(t) + numpy.random.normal(0, 0.1, t.size)

    return y



def smoothing_py(data, smooth_data):
    smooth_data[:] = 0.5*(data[:-2] + data[2:])



if __name__ == '__main__':
    n = 1000
    data = noisy_data(n)
    smooth_data = numpy.zeros(n-1)

    #smoothing_py(data, smooth_data)
    smoothing_f90(data, smooth_data, n)

    plt.ion()
    plt.plot(data, label='noisy')
    plt.plot(smooth_data, label='smooth')
    plt.legend()
    plt.show()
    raw_input()
    smoothing_noise()
