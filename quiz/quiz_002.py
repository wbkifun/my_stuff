'''
피보나치 수열의 각 항은 바로 앞의 항 두 개를 더한 것이 됩니다.
1과 2로 시작하는 경우 이 수열은 아래와 같습니다.
1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
짝수이면서 4백만 이하인 모든 항을 더하면 얼마가 될까요?
'''

from __future__ import print_function, division
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal




def fibonacci():
    '''
    Generate a Fibonacci sequence
    '''
    seq1, seq2 = 1, 2

    while True:
        yield seq1
        seq1, seq2 = seq2, seq1 + seq2




def sum_fibonacci_under(number):
    '''
    number보다 작은 Fibonacci 수열 더하기
    '''

    summed = 0

    # number보다 작은 Fibonacci 수열
    for fib in fibonacci():
        if fib > number:
            break
        elif fib%2 == 0:
            summed += fib

    return summed




def main():
    '''
    main()
    '''

    #
    # fibonacci() 함수 테스트
    #
    fib_list = list()
    for fib in fibonacci():
        if fib > 89:
            break
        else:
            fib_list.append(fib)
    a_equal(fib_list, [1, 2, 3, 5, 8, 13, 21, 34, 55, 89])

    #
    # sum_fibonacci_under() 함수 테스트
    #
    equal(sum_fibonacci_under(89), sum([2, 8, 34]))

    #
    # 문제 풀이
    #
    summed = sum_fibonacci_under(4000000)
    print('짝수이면서 4백만 이하인 피보나치 수열의 합?', summed)




if __name__ == '__main__':
    main()
