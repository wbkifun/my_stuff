#------------------------------------------------------------------------------
# filename  : find_optimal_partition.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.3.21     start
#
#
# description: 
#   Find optimal numbers of partition for cubed-sphere grid
#------------------------------------------------------------------------------




def factors(n):
    result = set()
    for i in range(1, int(n**0.5)+1):
        div, mod = divmod(n,i)
        if mod == 0:
            result |= {i, div}

    return result



if __name__ == '__main__':
    for ne in [30, 60, 120, 240, 480]:
        p_list = sorted(factors(ne))
        print("ne={:>3d}, factors={}".format(ne, p_list))

        for p in p_list:
            print("p={:>3d}, p2={:>6d}, nproc={:>7d}, elem/rank={:6.0f}".format(p, p*p, p*p*6, (ne*ne)/(p*p)))

        print()
