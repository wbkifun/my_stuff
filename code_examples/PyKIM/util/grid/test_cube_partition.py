#------------------------------------------------------------------------------
# filename  : test_cube_partition.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2015.9.8   revise
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
from numpy.testing import assert_equal as equal
from numpy.testing import assert_array_equal as a_equal
from numpy.testing import assert_array_almost_equal as aa_equal
from nose.tools import raises, ok_




def test_set_factor_list():
    '''
    factor_list: ne=30, nproc=1, 6*2*2, 6*3*3
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=2*3*5, nproc=1)
    a_equal(cube.factor_list, [2,3,5])

    cube = CubePartition(ne=2*3*5, nproc=6*2*2)
    a_equal(cube.factor_list, [3,5,2])

    cube = CubePartition(ne=2*3*5, nproc=6*3*3)
    a_equal(cube.factor_list, [2,5,3])




@raises(AssertionError)
def test_set_factor_list_exception():
    '''
    factor_list exception: ne=3*7, nproc=1
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=3*7, nproc=1)




def test_set_sfc():
    '''
    sfc: ne=4, 6, 10
    '''
    from cube_partition import CubePartition

    cube = CubePartition(ne=2*2, nproc=1)
    expect = np.array([ \
            [ 1, 4, 5, 6], \
            [ 2, 3, 8, 7], \
            [15,14, 9,10], \
            [16,13,12,11]], 'i4')
    a_equal(cube.sfc, expect)


    cube = CubePartition(ne=2*3, nproc=1)
    expect = np.array([ \
            [ 1, 2,15,16,17,18], \
            [ 4, 3,14,13,20,19], \
            [ 5, 8, 9,12,21,22], \
            [ 6, 7,10,11,24,23], \
            [35,34,31,30,25,26], \
            [36,33,32,29,28,27]], 'i4') 
    a_equal(cube.sfc, expect)

    cube = CubePartition(ne=2*5, nproc=1)
    expect = np.array([ \
            [ 1, 2,31,32,33,36,37,40,41,42], \
            [ 4, 3,30,29,34,35,38,39,44,43], \
            [ 5, 6,27,28,23,22,51,50,45,46], \
            [ 8, 7,26,25,24,21,52,49,48,47], \
            [ 9,12,13,16,17,20,53,56,57,58], \
            [10,11,14,15,18,19,54,55,60,59], \
            [95,94,91,90,79,78,75,74,61,62], \
            [96,93,92,89,80,77,76,73,64,63], \
            [97,98,87,88,81,82,71,72,65,66], \
           [100,99,86,85,84,83,70,69,68,67]], 'i4') 
    a_equal(cube.sfc, expect)




def test_elem_gseq():
    '''
    elem_gseq: ne=3, nproc=1
    '''
    from cube_partition import CubePartition

    ne = 3
    expect_p1 = np.array( \
            [[ 1, 4, 5], \
             [ 2, 3, 6], \
             [ 9, 8, 7]])

    expect_p2 = np.array( \
            [[10,11,18], \
             [13,12,17], \
             [14,15,16]])

    expect_p3 = np.array( \
            [[50,51,52], \
             [49,48,53], \
             [46,47,54]])

    expect_p4 = np.array( \
            [[34,33,32], \
             [35,30,31], \
             [36,29,28]])

    expect_p5 = np.array( \
            [[45,38,37], \
             [44,39,40], \
             [43,42,41]])

    expect_p6 = np.array( \
            [[27,26,25], \
             [20,21,24], \
             [19,22,23]])


    #np.set_printoptions(threshold=np.nan)
    cube = CubePartition(ne, nproc=1)
    a_equal(cube.elem_gseq[0,:,:], expect_p1)
    a_equal(cube.elem_gseq[1,:,:], expect_p2)
    a_equal(cube.elem_gseq[2,:,:], expect_p3)
    a_equal(cube.elem_gseq[3,:,:], expect_p4)
    a_equal(cube.elem_gseq[4,:,:], expect_p5)
    a_equal(cube.elem_gseq[5,:,:], expect_p6)




def test_elem_proc_3_4():
    '''
    elem_proc: ne=3, nproc=4
    '''
    from cube_partition import CubePartition

    expect_p1 = np.array( \
            [[ 0, 0, 0], \
             [ 0, 0, 0], \
             [ 0, 0, 0]])

    expect_p2 = np.array( \
            [[ 0, 0, 1], \
             [ 0, 0, 1], \
             [ 0, 1, 1]])

    expect_p3 = np.array( \
            [[ 3, 3, 3], \
             [ 3, 3, 3], \
             [ 3, 3, 3]])

    expect_p4 = np.array( \
            [[ 2, 2, 2], \
             [ 2, 2, 2], \
             [ 2, 2, 1]])

    expect_p5 = np.array( \
            [[ 3, 2, 2], \
             [ 3, 2, 2], \
             [ 3, 3, 2]])

    expect_p6 = np.array( \
            [[ 1, 1, 1], \
             [ 1, 1, 1], \
             [ 1, 1, 1]])

    cube = CubePartition(ne=3, nproc=4)
    a_equal(cube.nelems, [14, 14, 13, 13])
    a_equal(cube.elem_proc[0,:,:], expect_p1)
    a_equal(cube.elem_proc[1,:,:], expect_p2)
    a_equal(cube.elem_proc[2,:,:], expect_p3)
    a_equal(cube.elem_proc[3,:,:], expect_p4)
    a_equal(cube.elem_proc[4,:,:], expect_p5)
    a_equal(cube.elem_proc[5,:,:], expect_p6)




def test_set_elem_proc_3_7():
    '''
    elem_proc: ne=3, nproc=7
    '''
    from cube_partition import CubePartition

    expect_p1 = np.array( \
            [[ 0, 0, 0], \
             [ 0, 0, 0], \
             [ 1, 0, 0]])

    expect_p2 = np.array( \
            [[ 1, 1, 2], \
             [ 1, 1, 2], \
             [ 1, 1, 1]])

    expect_p3 = np.array( \
            [[ 6, 6, 6], \
             [ 6, 6, 6], \
             [ 5, 5, 6]])

    expect_p4 = np.array( \
            [[ 4, 4, 3], \
             [ 4, 3, 3], \
             [ 4, 3, 3]])

    expect_p5 = np.array( \
            [[ 5, 4, 4], \
             [ 5, 4, 4], \
             [ 5, 5, 5]])

    expect_p6 = np.array( \
            [[ 3, 3, 3], \
             [ 2, 2, 2], \
             [ 2, 2, 2]])

    cube = CubePartition(ne=3, nproc=7)
    a_equal(cube.nelems, [8, 8, 8, 8, 8, 7, 7])
    a_equal(cube.elem_proc[0,:,:], expect_p1)
    a_equal(cube.elem_proc[1,:,:], expect_p2)
    a_equal(cube.elem_proc[2,:,:], expect_p3)
    a_equal(cube.elem_proc[3,:,:], expect_p4)
    a_equal(cube.elem_proc[4,:,:], expect_p5)
    a_equal(cube.elem_proc[5,:,:], expect_p6)




def test_set_elem_proc_4_5():
    '''
    elem_proc(HOMME): ne=4, nproc=5
    '''
    from cube_partition import CubePartition

    expect_p1 = np.array( \
            [[ 0, 0, 0, 0], \
             [ 0, 0, 0, 0], \
             [ 0, 0, 0, 0], \
             [ 0, 0, 0, 0]])

    expect_p2 = np.array( \
            [[ 1, 1, 0, 0], \
             [ 1, 1, 0, 0], \
             [ 1, 1, 1, 1], \
             [ 1, 1, 1, 1]])

    expect_p3 = np.array( \
            [[ 4, 4, 4, 4], \
             [ 4, 4, 4, 4], \
             [ 4, 4, 4, 4], \
             [ 4, 4, 4, 4]])

    expect_p4 = np.array( \
            [[ 3, 3, 2, 2], \
             [ 3, 3, 2, 2], \
             [ 3, 2, 2, 2], \
             [ 3, 2, 2, 2]])

    expect_p5 = np.array( \
            [[ 3, 3, 3, 3], \
             [ 3, 3, 3, 3], \
             [ 4, 4, 3, 3], \
             [ 4, 3, 3, 3]])

    expect_p6 = np.array( \
            [[ 2, 2, 2, 2], \
             [ 2, 2, 2, 2], \
             [ 1, 2, 1, 1], \
             [ 1, 1, 1, 1]])

    cube = CubePartition(ne=4, nproc=5, homme_style=True)
    a_equal(cube.nelems, [20, 19, 19, 19, 19])
    a_equal(cube.elem_proc[0,:,:], expect_p1)
    a_equal(cube.elem_proc[1,:,:], expect_p2)
    a_equal(cube.elem_proc[2,:,:], expect_p3)
    a_equal(cube.elem_proc[3,:,:], expect_p4)
    a_equal(cube.elem_proc[4,:,:], expect_p5)
    a_equal(cube.elem_proc[5,:,:], expect_p6)




def test_set_elem_proc_30_16():
    '''
    elem_proc(HOMME): ne=30, nproc=16
    '''
    from glob import glob
    import re
    import netCDF4 as nc
    from cube_mpi import CubeGridMPI

    ne, ngq = 30, 4
    nproc = 16

    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank=0, homme_style=True)
    my_elem_gid = cubegrid.local_gids[::ngq*ngq]//(ngq*ngq) + 1
    homme_elem_gid = np.array([275, 276, 301, 302, 303, 304, 305, 306, 331, 332, 333, 334, 335, 336, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 751, 752, 753, 754, 755, 756, 757, 758 ,759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888], 'i4') 
    a_equal(my_elem_gid, homme_elem_gid)


    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank=1, homme_style=True)
    my_elem_gid = cubegrid.local_gids[::ngq*ngq]//(ngq*ngq) + 1
    homme_elem_gid = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 295, 296, 297, 298, 299, 300, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 325, 326, 327, 328, 329, 330, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 355, 356, 357, 358, 359, 360], 'i4') 
    a_equal(my_elem_gid, homme_elem_gid)


    cubegrid = CubeGridMPI(ne, ngq, nproc, myrank=2, homme_style=True)
    my_elem_gid = cubegrid.local_gids[::ngq*ngq]//(ngq*ngq) + 1
    homme_elem_gid = np.array([263, 264, 293, 294, 323, 324, 353, 354, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 1573, 1574, 1575, 1603, 1604, 1605, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, ], 'i4') 
    a_equal(my_elem_gid, homme_elem_gid)


    fpath_list = glob('./KIM_lid_gid_ne30_nproc16/nproc16_rank*.nc')
    for fpath in fpath_list:
        ncf = nc.Dataset(fpath, 'r', format='NETCDF3_CLASSIC')
        local_element_size = len( ncf.dimensions['local_element_size'] )
        lid_elements = ncf.variables['lid_elements'][:]
        gid_elements = ncf.variables['gid_elements'][:]

        rank = int( re.search('rank([0-9]+).nc',fpath).group(1) )
        cubegrid = CubeGridMPI(ne, ngq, nproc, rank, homme_style=True)
        my_elem_gid = cubegrid.local_gids[::ngq*ngq]//(ngq*ngq) + 1

        a_equal(my_elem_gid, gid_elements)
