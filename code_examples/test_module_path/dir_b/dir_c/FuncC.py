import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
up_dpath = dirname(current_dpath)
up2_dpath = dirname(up_dpath)
sys.path.extend([current_dpath,up_dpath,up2_dpath])

from dir_a.FuncA import fa
from dir_b.FuncB import fb


def fc():
    print('fc()')
    print('call fa()')
    fa()
    print('call fb()')
    fb()
