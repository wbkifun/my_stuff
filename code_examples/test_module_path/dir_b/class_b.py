import sys
from os.path import abspath, dirname
current_dpath = dirname(abspath(__file__))
up_dpath = dirname(current_dpath)
sys.path.extend([current_dpath,up_dpath])

from dir_a.class_a import CA


class CB(CA):
    def __init__(self, b):
        super(CB, self).__init__(b)
