from __future__ import division
import numpy


class MyClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.c = a + b
        self.d = a * b


    def divide(self):
        a = self.a
        b = self.b

        return a/b



if __name__ == '__main__':
    mc = MyClass(1.2, 3.5)
    print mc.c
    print mc.d
    print mc.divide()

    print '\nget all member variables'
    member = mc.__dict__
    for key, val in member.items():
        print key, val
