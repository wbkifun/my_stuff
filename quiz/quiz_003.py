'''
어떤 수를 소수의 곱으로만 나타내는 것을 소인수분해라고 하고,
그 소수들을 그 수의 소인수라고 하죠.
예를 들면 13195의 소인수는 5, 7, 13, 29 입니다.
600851475143 의 소인수 중에서 가장 큰 수를 구하세요.
'''

from __future__ import print_function, division
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal




def prime_factors(number):
    '''
    Find prime factors of a given number
    '''

    seq = 2
    number2 = number
    p_factors = list()

    while seq*seq <= number2:
        if number2%seq != 0:
            seq += 1
        else:
            number2 //= seq
            p_factors.append(seq)

    if number2 > 1:
        p_factors.append(number2)

    return p_factors




def main():
    '''
    main()
    '''

    #
    # prime_factors() 함수 테스트
    #
    a_equal(prime_factors(13195), [5, 7, 13, 29])

    #
    # 문제 풀이
    #
    number = 600851475143
    p_factors = prime_factors(number)
    print('{} 의 소인수들: {}'.format(number, p_factors))
    print('{} 의 최대소인수: {}'.format(number, p_factors[-1]))




if __name__ == '__main__':
    main()
