# http://www.pythonchallenge.com/pc/def/map.html
#
# K -> M
# O -> Q
# E -> G
#
# everybody thinks twice before solving this.
#
# solution : http://www.pythonchallenge.com/pc/def/ocr.html


import string



ss = "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."



def shift_ascii(c, shift):
    return chr(ord(c) + shift)



def try1():
    print [shift_ascii(c, 2) for c in ['K','O','E']]
    #ss2 = [shift_ascii(c, 2) for c in ss if c != ' ']

    ss2 = ''
    for c in ss:
        if c in [' ', '.', '\'', '(', ')']:
            ss2 += c
        elif c == 'y':
            ss2 += 'a'
        elif c == 'z':
            ss2 += 'b'
        else:
            ss2 += shift_ascii(c,2)

    print ''.join(ss2)



def try2():
    intab = ''.join([chr(i) for i in xrange(97,123)])
    outab = ''.join([chr(i) for i in range(99,123)+[97,98]])
    trantab = string.maketrans(intab,outab)
    print ss.translate(trantab)
    print 'map -> ', ''.join([shift_ascii(c, 2) for c in 'map'])



if __name__ == '__main__':
    print 'try1'
    try1()
    print '-'*80

    print 'try2'
    try2()
