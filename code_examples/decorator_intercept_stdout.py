import io
import sys 


def intercept_stdout(func):
    def wrapper(*args, **kwargs):
        capturer = io.StringIO()
        old_stdout, sys.stdout = sys.stdout, capturer

        ret = func(*args, **kwargs)

        sys.stdout = old_stdout
        output = capturer.getvalue().rstrip('\n')

        return ret, output

    return wrapper


@intercept_stdout
def myfunc(a, b=0.7):
    print('a=',a)
    print('b=',b)
    print('c={}'.format(a+b))
    return a+b


if __name__ == '__main__':
    print('Before')
    ret, output = myfunc(10.5)
    print('After')

    print("ret={}, output='{}'".format(ret,output))
