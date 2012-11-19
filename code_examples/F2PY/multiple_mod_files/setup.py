from numpy.distutils.core import setup, Extension

setup( 
    ext_modules = [Extension('sumavg', \
            ['summod.f90', 'avgmod.f90', 'sumavgmod.f90']) ]
)
