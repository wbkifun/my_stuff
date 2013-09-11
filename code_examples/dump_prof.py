import pstats
import sys

'''
$ python -m cProfile -o out_filename run.py
'''

fpath = sys.argv[1]
st = pstats.Stats(fpath)
dr = st.strip_dirs()
sort = dr.sort_stats("cumulative")
sort.print_stats()
