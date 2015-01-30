import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('echo', help='echo the string you use here')
parser.add_argument('square', type=int, help='display a square of a given number')
parser.add_argument('ne', type=int, help='number of element')
parser.add_argument('ngq', type=int, nargs='?', default=4, help='number of Gauss quadrature points')
#parser.add_argument('-v','--verbose', help='increase output verbosity', action='store_true')
parser.add_argument('-v','--verbose', type=int, choices=[0,1,2], default=2, help='increase output verbosity')

args = parser.parse_args()

print (args.echo)
print 'ne=%d, ngq=%d'%(args.ne, args.ngq)

answer = args.square**2
if args.verbose == 2:
    print 'The square of {} equals {}.'.format(args.square, answer)
elif args.verbose == 1:
    print '{}^2 == {}'.format(args.square, answer)
else:
    print answer
