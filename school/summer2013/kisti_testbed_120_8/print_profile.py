#!/usr/bin/env python


import pstats
stat = pstats.Stats('out2.prof')
stat.sort_stats('cumulative').print_stats(30)
