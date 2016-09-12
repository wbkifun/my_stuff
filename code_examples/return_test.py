def A(a, **kwargs):
    return globals()['B'](a, **kwargs)



class B(object):
    def __init__(self, b, c='all'):
        self.b = b
        self.c = c




if __name__ == '__main__':
    aa = A(3.4)
    print('aa.b', aa.b)
    print('aa.c', aa.c)

    bb = A(3.4, c='none')
    print('bb.b', bb.b)
    print('bb.c', bb.c)
