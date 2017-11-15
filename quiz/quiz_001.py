'''
10보다 작은 자연수 중에서 3 또는 5의 배수는 3,5,6,9 이고, 이것을 모두 더하면 23입니다.
1000보다 작은 자연수 중에서 3 또는 5의 배수를 모두 더하면 얼마일까요?
'''

from __future__ import print_function, division
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal




def sum_multiple_under(number):
    '''
    number보다 작은 자연수 중에서
    3 또는 5의 배수 곱하기
    '''

    summed = 0

    # number보다 작은 자연수
    for i in range(1, number):
        # 3 또는 5의 배수 확인
        if i%3 == 0 or i%5 == 0:
            summed += i

    return summed




def main():
    '''
    main()
    '''

    #
    # 함수 테스트
    #
    equal(sum_multiple_under(10), 23)

    #
    # 문제 풀이
    #
    summed = sum_multiple_under(1000)
    print('1000보다 작은 자연수 중에서 3 또는 5의 배수의 합?', summed)




if __name__ == '__main__':
    main()
