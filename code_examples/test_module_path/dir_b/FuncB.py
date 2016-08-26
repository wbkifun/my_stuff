import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
up_dpath = dirname(current_dpath)
sys.path.extend([current_dpath,up_dpath])

from dir_a.FuncA import fa


def fb():
    print('fb()')
    print('call fa()')
    fa()
