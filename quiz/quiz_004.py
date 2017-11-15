'''
앞에서부터 읽을 때나 뒤에서부터 읽을 때나 모양이 같은 수를 대칭수(palindrome)라고 합니다.
두 자리 수를 곱해 만들 수 있는 대칭수 중 가장 큰 수는 9009 (91x99)입니다.
세 자리 수를 곱해서 만들 수 있는 가장 큰 대칭수는 얼마일까요?
'''

from __future__ import print_function, division
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal




def find_max_palindrome(num_digit):
    '''
    Find a maximum palindrome
    '''

    max_limit = int('1'*num_digit)*9
    min_limit = int('1'+'0'*(num_digit-1))

    palindrome_dict = dict()
    for i in range(max_limit, min_limit, -1):
        for j in range(max_limit, min_limit, -1):
            num = i*j
            snum = str(num)

            if snum == snum[::-1]:
                palindrome_dict[num] = (i, j)

    max_num = max(palindrome_dict)
    factor1, factor2 = palindrome_dict[max_num]

    return max_num, factor1, factor2




def main():
    '''
    main()
    '''

    #
    # 두 자리 수로 만들어진 가장 큰 대칭수 9009(91x99) 테스트
    #
    a_equal(find_max_palindrome(2), (9009, 91, 99))

    #
    # 문제 풀이
    #
    num_digit = 3
    p_factors = find_max_palindrome(num_digit)
    number, factor1, factor2 = find_max_palindrome(3)
    print('{} 자리수로 만들어진 최대 대칭수: {} ({}x{})'.format( \
            num_digit, number, factor1, factor2))




if __name__ == '__main__':
    main()
