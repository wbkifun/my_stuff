import traceback


def get_funcname():
    stack = traceback.extract_stack()
    filename, codeline, funcname, text = stack[-2]

    return funcname



def my_func1():
    print get_funcname()


def my_func2():
    print get_funcname()



if __name__ == '__main__':
    my_func1()
    my_func2()
