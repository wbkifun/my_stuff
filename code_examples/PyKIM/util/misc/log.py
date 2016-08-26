#------------------------------------------------------------------------------
# filename  : log.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.29   start
#
# Customized logging
#------------------------------------------------------------------------------

import logging

from mpi4py import MPI 

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
myrank = comm.Get_rank()




class FileFilter(logging.Filter):
    def __init__(self, filenames=None):
        self.filenames = filenames

    def filter(self, record):
        if self.filenames is None:
            allow = True
        else:
            allow = record.filename not in self.filenames

        return allow



logger = logging.getLogger('rank{}'.format(myrank))
formatter = logging.Formatter('[%(name)s][%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s] > %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addFilter(logging.Filter('rank0'))

logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

#logger.addFilter(FileFilter(['device.py']))
