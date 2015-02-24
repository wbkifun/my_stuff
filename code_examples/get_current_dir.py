import os


# the view of this file
print __file__
print os.path.realpath(__file__)
print os.path.dirname(os.path.realpath(__file__))


# the view of execution point
print os.getcwd()
