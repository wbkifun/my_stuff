#!/usr/bin/env python

import numpy as np


class AA:
	def __init__(s, name, n, **kargs):
		s.n = n
		s.ez = np.zeros(s.n, 'f')
		s.hy = np.ones(s.n, 'f')

		print name
		print getattr(s, 'ez')

		if kargs.has_key('bb'):
			if kargs['bb'] == 'on':
				s.AAA = AA('sub', 5)


aa = AA('main', 10, bb='on')
