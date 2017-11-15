'''
1~10 사이의 어떤 수로도 나누어 떨어지는 가장 작은 수는 2520입니다.
1~20 사이의 어떤 수로도 나누어 떨어지는 가장 작은 수는 얼마일까요?
'''

from __future__ import print_function, division
#from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal




def great_common_divisor(num1, num2):
    '''
    두 정수 a, b의 최대공약수 G

    유클리드 호제법 (Euclidean algorithm)
    a, b의 최대공약수를 구하는 함수를 g(a,b)라고 하자.
    만약 a mod b = 0 이면, g(a,b) = b 이다.
    만약 a mod b != 0 이면, g(a,b) = g(b,a mod b) 이다.
    '''

    if num1%num2 == 0:
        return num2
    else:
        return great_common_divisor(num2, num1%num2)



def least_common_multiple(num_list):
    '''
    두 정수 a, b의 최대공약수를 G 라고 하면,
    두 정수의 최소공배수는 (a*b)/G 가 된다.
    <증명>
    a, b의 최대공약수가 G이면,
    a = G*x, b = G*y 이고 x, y는 서로 소가 된다.
    따라서 최대공배수는 G*x*y = (a*b)/G 가 된다.
    '''

    lcm = 1
    for num in num_list:
        gcd = great_common_divisor(lcm, num)
        lcm = (lcm*num)//gcd

    return lcm



def main():
    '''
    main()
    '''

    #
    # 1~10 사이의 어떤 수로도 나누어 떨어지는 가장 작은 수 2520 테스트
    #
    a_equal(least_common_multiple(range(1, 11)), 2520)

    #
    # 문제 풀이
    #
    lcm = least_common_multiple(range(1, 21))
    print('1~20 사이의 어떤 수로도 나누어 떨어지는 가장 작은 수: {}'.format(lcm))




if __name__ == '__main__':
    main()
