# Author: Raymond Hettingers
# refer to http://rosettacode.org/wiki/N-queens_problem#Python


from itertools import permutations

n = 8
cols = range(n+1)
nsol = 0
for vec in permutations(cols):
    if n + 1 == len(set(vec[i]+i for i in cols)) \
             == len(set(vec[i]-i for i in cols)):
        #print (vec)
        nsol += 1
