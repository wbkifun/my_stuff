#-*- coding: UTF-8 -*-

#------------------------------------------------------------------------------
# filename: grouping.py
# written by Ki-Hwan Kim
#
# changelog
# 2013.1.12  start
# 2013.6.4   update database
# 2013.8.2   update database
#------------------------------------------------------------------------------

from __future__ import division
import numpy
import random
import sys
import time


whole_members = {
        '박훈': ['남', '단장', ''], \
        '송인선': ['남', '개발본부장', ''], \
        '정종희': ['남', '지원본부장', ''], \
        '오태진': ['남', '개발본부', '역학코어팀'], \
        '이태형': ['남', '개발본부', '역학코어팀'], \
        '최석진': ['남', '개발본부', '역학코어팀'], \
        '강신후': ['남', '개발본부', '역학코어팀'], \
        '박자린': ['여', '개발본부', '역학코어팀'], \
        '윤선희': ['여', '개발본부', '역학코어팀'], \
        '남현'  : ['여', '개발본부', '역학코어팀'], \
        '김상일': ['남', '개발본부', '자료동화팀'], \
        '송효종': ['남', '개발본부', '자료동화팀'], \
        '권하택': ['여', '개발본부', '자료동화팀'], \
        '강전호': ['남', '개발본부', '자료동화팀'], \
        '이시혜': ['여', '개발본부', '자료동화팀'], \
        '정병주': ['남', '개발본부', '자료동화팀'], \
        '전형욱': ['남', '개발본부', '자료동화팀'], \
        '강지순': ['여', '개발본부', '자료동화팀'], \
        '권지혜': ['여', '개발본부', '자료동화팀'], \
        '조영순': ['여', '개발본부', '자료동화팀'], \
        '김주혜': ['여', '개발본부', '자료동화팀'], \
        '박종임': ['여', '개발본부', '자료동화팀'], \
        '임주연': ['여', '개발본부', '자료동화팀'], \
        '하수진': ['여', '개발본부', '자료동화팀'], \
        '진경'  : ['여', '개발본부', '물리모수화팀'], \
        '김소영': ['여', '개발본부', '물리모수화팀'], \
        '이준석': ['남', '개발본부', '물리모수화팀'], \
        '한지영': ['여', '개발본부', '물리모수화팀'], \
        '배수야': ['여', '개발본부', '물리모수화팀'], \
        '최현주': ['여', '개발본부', '물리모수화팀'], \
        '최인진': ['여', '개발본부', '물리모수화팀'], \
        '이은희': ['여', '개발본부', '물리모수화팀'], \
        '조경미': ['여', '개발본부', '물리모수화팀'], \
        '전혜연': ['여', '개발본부', '물리모수화팀'], \
        '신설은': ['여', '시스템본부', '시스템개발팀'], \
        '주동찬': ['남', '시스템본부', '시스템개발팀'], \
        '김정한': ['남', '시스템본부', '시스템개발팀'], \
        '김기환': ['남', '시스템본부', '시스템개발팀'], \
        '전상윤': ['남', '시스템본부', '시스템개발팀'], \
        '권인혁': ['남', '시스템본부', '시스템개발팀'], \
        '이영수': ['남', '시스템본부', '시스템개발팀'], \
        '김형주': ['여', '시스템본부', '시스템개발팀'], \
        '정길란': ['여', '시스템본부', '모델검증팀'], \
        '김태균': ['남', '시스템본부', '모델검증팀'], \
        '강정윤': ['여', '시스템본부', '모델검증팀'], \
        '채정효': ['남', '시스템본부', '모델검증팀'], \
        '설경희': ['여', '시스템본부', '모델검증팀'], \
        '이주원': ['여', '시스템본부', '모델검증팀'], \
        '진선화': ['여', '시스템본부', '모델검증팀'], \
        '장동일': ['남', '시스템본부', '모델검증팀'], \
        '이상훈': ['남', '시스템본부', '모델검증팀'], \
        '김윤항': ['남', '시스템본부', '현업화지원팀'], \
        '백지욱': ['남', '시스템본부', '현업화지원팀'], \
        '정래윤': ['남', '시스템본부', '현업화지원팀'], \
        '손호진': ['남', '시스템본부', '현업화지원팀'], \
        '연승민': ['남', '시스템본부', '현업화지원팀'], \
        '박지현': ['여', '시스템본부', '현업화지원팀'], \
        '엄미진': ['여', '시스템본부', '현업화지원팀'], \
        '최정민': ['남', '지원본부', '연구기획팀'], \
        '김민희': ['여', '지원본부', '연구기획팀'], \
        '박현민': ['남', '지원본부', '연구기획팀'], \
        '김재헌': ['남', '지원본부', '운영지원팀'], \
        '이아름이': ['여', '지원본부', '운영지원팀'], \
        '한유진': ['여', '지원본부', '운영지원팀'], \
        '이광엽': ['남', '지원본부', '운영지원팀'], \
        '정기섭': ['남', '지원본부', '운영지원팀'], \
        '최이태': ['남', '지원본부', '대외협럭팀'], \
        '구은성': ['여', '지원본부', '대외협럭팀'], \
        '김민영': ['여', '지원본부', '대외협럭팀'], \
        '고연정': ['여', '지원본부', '대외협력팀'], \
        '나지성': ['남', '시스템본부', '모델검증팀'] }

position_groups = [ \
        ['김태균', '설경희', '이주원', '강정윤', '채정효', '진선화'], \
        ['권인혁', '이상훈', '엄미진', '김정한', '전상윤', '이영수'], \
        ['김기환', '주동찬', '손호진', '김윤항', '정래윤', '손호진'], \
        ['구은성', '고연정', '김민영', '김민희', '박현민', '이광엽', '정기섭', \
         '이아름이', '한유진', '최이태', '최정민', '김재헌'], \
        ['배수야', '조경미', '박지현', '최현주', '한지영'], \
        ['김소영', '이준석', '전혜연', '최인진', '이은희'], \
        ['강신후', '남현', '박자린', '윤선희', '최석진'], \
        ['강전호', '권하택', '이시혜'], \
        ['김주혜', '전형욱', '박종임', '강지순', '권지혜'], \
        ['정병주', '조영순', '임주연', '하수진'] ]

previous_groupss = [ \
        [['강전호', '이태형', '강신후', '주동찬', '권지혜', '임주연', '진경'], \
         ['김기환', '박현민', '송인선', '나지성', '김민영', '조영순', '한지영'], \
         ['김민희', '김재헌', '최석진', '손호진', '구은성', '김소영', '진선화'], \
         ['김태균', '최정민', '오태진', '김정한', '권하택', '최현주', '강정윤'], \
         ['송효종', '최이태', '이준석', '백지욱', '채정효', '이시혜', '배수야', \
          '신호정'], \
         ['이아름이', '이광엽', '정병주', '전상윤', '한유진', '신현진', '조경미'] \
        ], \
        [['강전호', '박현민', '김상일', '백지욱', '나지성', '김민영', '임주연', \
          '김소영', '유선희', '이영수'], \
         ['김기환', '최정민', '최석진', '주동찬', '구은성', '조영순', '강정윤', \
          '신호정', '김윤항', '신설은'], \
         ['김민희', '김재헌', '정병주', '이준석', '박자린', '권지혜', '배수야', \
          '진선화', '박종임'], \
         ['김태균', '이태형', '이광엽', '김정한', '채정효', '신현진', '이시혜', \
          '조경미'], \
         ['송효종', '최이태', '강신후', '송인선', '정래윤', '한유진', '진경', \
          '최현주', '김주혜'], \
         ['이아름이', '오태진', '신동욱', '전상윤', '손호진', '고연정', '권하택', \
          '한지영', '이주원'] \
        ], \
        [['박자린', '이광엽', '나지성', '전형욱', '정래윤', '김소영', '권하택', '이시혜'], \
         ['강신후', '김민희', '김정한', '한지영', '백지욱', '강지순', '신호정', '김재헌'], \
         ['최석진', '송인선', '정기섭', '정병주', '이주원', '조경미', '권지혜', '정길란'], \
         ['윤선희', '최정민', '전상윤', '이준석', '진선화', '김주혜', '김윤항', '엄미진'], \
         ['이태형', '신설은', '박현민', '최현주', '채정효', '김상일', '조영순'], \
         ['오태진', '설경희', '김민영', '신현진', '주동찬', '한유진', '강전호', '하수진'], \
         ['진경', '손호진', '이아름이', '박종임', '강정윤', '김기환', '최이태', '박지현'], \
         ['송효종', '이영수', '구은성', '김태균', '최인진', '고연정', '임주연', '전혜연'] \
        ], \
        [['오태진', '이영수', '김민희', '김주혜', '정길란', '이광엽', '손호진'], \
         ['권하택', '구은성', '백지욱', '진경', '정병주', '박현민', '전상윤'], \
         ['최정민', '한지영', '정래윤', '박종임', '설경희', '조경미', '강전호'], \
         ['강정윤', '고연정', '이태형', '엄미진', '하수진', '송인선', '강신후'], \
         ['김태균', '김상일', '한유진', '이준석', '강지순', '신설은', '김소영'], \
         ['김정한', '최이태', '최석진', '전혜연', '김민영', '박자린', '조영순'], \
         ['최현주', '주동찬', '이아름이', '나지성', '윤선희', '송효종', '권지혜'], \
         ['채정효', '정기섭', '전형욱', '박지현', '최인진', '진선화'], \
         ['이시혜', '김재헌', '김기환', '이주원', '임주연', '김윤항'] \
        ], \
        [['김주혜', '김정한', '박현민', '최인진', '장동일', '주동찬', '박자린'], \
         ['정병주', '나지성', '최이태', '이태형', '강지순', '이주원', '김소영'], \
         ['전상윤', '강전호', '정기섭', '전혜연', '강정윤', '이시혜', '손호진'], \
         ['이준석', '이아름이', '정래윤', '임주연', '최석진', '조영순', '한지영'], \
         ['진선화', '이은희', '김민영', '강신후', '이광엽', '김기환', '조경미'], \
         ['김민희', '진경'  , '채정효', '윤선희', '고연정', '김상일', '최정민'], \
         ['김재헌', '정길란', '하수진', '박지현', '최현주', '백지욱', '김태균'], \
         ['권지혜', '구은성', '설경희', '권인혁', '엄미진', '전형욱', '오태진'], \
         ['박종임', '이영수', '한유진', '김윤항', '권하택', '송효종'] \
        ] ]

except_members = ['박훈'  , '정종희', '송인선',\
                  '김형주', '연승민', \
                  '신설은', '오태진', '김기환', '나지성']

manager_group = ['진경', '신설은', '송효종', '오태진', '김윤항', \
                 '최정민', '최이태', '김재헌']

couple_groups = [['정병주', '김주혜'], ['김상일', '박지현']]



#==============================================================================
#------------------------------------------------------------------------------
# input
#------------------------------------------------------------------------------
print('BBL조 편성을 시작합니다.\n')
print('팀, 본부, 좌석 위치, 이전의 조편성, 남여 비율, 등을 고려하여')
print('평소에 만나기 힘든 사람들을 위주로 조를 편성합니다.\n')
while True:
    num_per_group = raw_input('조별 인원수를 입력해주세요 : ')
    try:
        num_per_group = int(num_per_group)
        if type(num_per_group) == int: break
    except:
        print('잘못된 입력입니다.\n')


#------------------------------------------------------------------------------
# allocate
#------------------------------------------------------------------------------
members = whole_members.copy()
for except_member in except_members:
    members.pop( except_member )

num_men, num_women = 0, 0 
for idx, (sex, department, team) in enumerate( members.values() ):
    if sex == '남': num_men += 1
    elif sex == '여': num_women += 1

N = len(members)
ngroup = N // num_per_group
num_remain = N % num_per_group
if num_remain < ngroup:
    ngroup += 1

name_list = members.keys()
random.shuffle( name_list )
nearness = numpy.zeros((N,N), 'i4')
groups = [list() for i in xrange(ngroup)]

ngroups = [N//ngroup for i in xrange(ngroup)]
for idx in xrange(N%ngroup):
    ngroups[idx] += 1

max_men = [num_men//ngroup for i in xrange(ngroup)]
for idx in xrange(num_men%ngroup):
    max_men[idx] += 1
max_women = [ngroups[i] - max_men[i] for i in xrange(ngroup)]
men = [0 for i in xrange(ngroup)]
women = [0 for i in xrange(ngroup)]


#------------------------------------------------------------------------------
# ask to continue
#------------------------------------------------------------------------------
print('\n현재 BBL 참여 인원은 %d 명(남 %d명, 여 %d명) 입니다.' % \
        (N, num_men, num_women))
print('총 %d 개의 조로 편성됩니다.' % ngroup)
print('조별 인원수 : %s' % ngroups)
is_go = raw_input('\n계속하시겠습니까?(Y/n) ' )
if is_go not in ['', 'Y', 'y']:
    print('\n프로그램을 마칩니다. 감사합니다.\n')
    sys.exit()


#------------------------------------------------------------------------------
# determine the nearness table
#------------------------------------------------------------------------------
for i in xrange(N):
    for j in xrange(N):
        name1, name2 = name_list[i], name_list[j]
        team1, team2 = members[name1][2], members[name2][2]
        department1, department2 = members[name1][1], members[name2][1]

        # couple?
        for c_group in couple_groups:
            if name1 in c_group and name2 in c_group: 
                nearness[i,j] += 100
                break

        # same previous_group?
        weight = 50
        for previous_groups in previous_groupss[::-1]:
            for p_group in previous_groups:
                if name1 in p_group and name2 in p_group: 
                    nearness[i,j] += weight
                    weight = weight//2
                    break

        # same team?
        if team1 == team2: 
            nearness[i,j] += 10

        # both manager?
        if name1 in manager_group and name2 in manager_group:
            nearness[i,j] += 10

        # near position?
        for p_group in position_groups:
            if name1 in p_group and name2 in p_group: 
                nearness[i,j] += 3
                break

        # same department?
        if department1 == department2: 
            nearness[i,j] += 2


#------------------------------------------------------------------------------
# generate the groups
#------------------------------------------------------------------------------
def find_name(avail_name_list, target_name_list, maxmin, sex=None):
    sum_nearness = numpy.zeros( len(avail_name_list) )

    for idx, name in enumerate(avail_name_list):
        ni = name_list.index(name)

        for target_name in target_name_list:
            nj = name_list.index( target_name )
            sum_nearness[idx] += nearness[ni,nj]

        print('%12s\t(%3d)\r' % (name, sum_nearness[idx])),
        sys.stdout.flush()
        time.sleep(0.02)     # time interval (sec)
            
    if maxmin == 'max': 
        idx = sum_nearness.argmax()
        name = avail_name_list[idx]

    elif maxmin == 'min': 
        if sex == None:
            argmin = sum_nearness.argmin()
            name = avail_name_list[argmin]
        else:
            while True:
                argmin = sum_nearness.argmin()
                name = avail_name_list[argmin]
                if members[name][0] != sex:
                    sum_nearness[argmin] = 1e8
                else:
                    break

    return name


#------------------------------------------------------------------------------
# find the first members
seed = numpy.random.randint(N)
avail_name_list = name_list[:]
first_name = avail_name_list[seed]

print('\n' + '-'*47)
print('먼저 각 조에 조원들을 한 명씩 배정합니다.')
print("\n행운의 첫 번째 사람은 '%s' 님이네요. ^^\n" % first_name)
print("'%s' 님을 포함해서 선택된 사람들과 자주 만나는 사람들을 각 조에 한 명씩 배정합니다.\n" % first_name)

for gi, group in enumerate(groups):
    if gi == 0:
        name = first_name

    else:
        target_name_list = [groups[gj][0] for gj in xrange(gi)]
        name = find_name(avail_name_list, target_name_list, 'max') 

    group.append(name)
    avail_name_list.remove(name)

    if members[name][0] == '남': men[gi] += 1
    else: women[gi] += 1
    print('%d 조 : %s          ' % (gi+1, name))

#------------------------------------------------------------------------------
# find the other members
print('\n' + '-'*47)
print('각 조에 나머지 조원들을 배정합니다.')
print('남여 비율을 고려하여 조원들과 만나기 힘든 사람들을 우선 검색합니다.')

while len(avail_name_list):
    print('')
    for gi, group in enumerate(groups):
        target_name_list = group
        if men[gi] == max_men[gi]:
            name = find_name(avail_name_list, target_name_list, 'min', '여') 
        elif women[gi] == max_women[gi]:
            name = find_name(avail_name_list, target_name_list, 'min', '남') 
        else:
            name = find_name(avail_name_list, target_name_list, 'min') 

        group.append(name)
        avail_name_list.remove(name)

        if members[name][0] == '남': men[gi] += 1
        else: women[gi] += 1
        print('%d 조 : %s          ' % (gi+1, name))

        if len(avail_name_list) == 0: break


#------------------------------------------------------------------------------
# print the groups
#------------------------------------------------------------------------------
print('\n'+'-'*47)
print(' Summary')
print(''+'-'*47)

for gidx, group in enumerate(groups):
    print('%d 조 : ' % (gidx+1)),

    for name in group:
        print('%s\t' % name),

    sum_nearness = numpy.zeros( len(group) )
    for idx, name in enumerate(group):
        group2 = group[:]
        group2.remove(name)
        ni = name_list.index(name)
        for target_name in group2:
            nj = name_list.index( target_name )
            sum_nearness[idx] += nearness[ni,nj]

    print('(%d)\n' % (sum_nearness.sum()//len(group)) )
