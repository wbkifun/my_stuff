#------------------------------------------------------------------------------
# filename  : human_readable_bytes.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.10.29    start
#
# refer to http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
#------------------------------------------------------------------------------

from math import log




def human_readable_bytes(n,pow=0,b=1024,u='B',pre=['']+[p+'i'for p in'KMGTPEZY']):
    r,f=min(int(log(max(n*b**pow,1),b)),len(pre)-1),'{:,.%if} %s%s'
    return (f%(abs(r%(-r-1)),pre[r],u)).format(n*b**pow/b**float(r))




if __name__ == '__main__':
    for byte in [32, 2015, 987654321, 9876543210, 987654321098765432109876543210]:
        print human_readable_bytes(byte)

    print human_readable_bytes(0.5, pow=2)
