#------------------------------------------------------------------------------
# filename  : log.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.29   start
#
# Customized logging
#------------------------------------------------------------------------------

from __future__ import division
import logging



class FileFilter(logging.Filter):
    def __init__(self, filenames=None):
        self.filenames = filenames

    def filter(self, record):
        if self.filenames is None:
            allow = True
        else:
            allow = record.filename not in self.filenames

        return allow



logger = logging.getLogger()
formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] > %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

#logger.addFilter(FileFilter(['device.py']))
