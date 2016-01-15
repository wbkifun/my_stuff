#------------------------------------------------------------------------------
# filename  : plot_remap_accuracy.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.1.14     start
#
# Description: 
#   Plot remapping accuracy of several methods
#   between Cubed-sphere and Latlon grids
#
# Resolutions
#   cube      latlon
#   ne30  <-> 180x360
#   ne60  <-> 360x720
#   ne120 <-> 720x1440
#
# Methods
#   Bilinear, V-GECoRe, RBF, Lagrange
#
# Test Function
#   Spherical harmonics Y(m=16,n=32)
#------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt



#-----------------------------------------------------------------------------
# Experimental results
#-----------------------------------------------------------------------------
ll2cs = {'ne30': {'Bilinear':(1.9798e-2, 2.0747e-2, 2.0620e-2), 
                  'V-GECoRe':(5.8266e-2, 5.7656e-2, 7.9632e-2),
                  'RBF'     :(3.6341e-3, 3.8378e-3, 4.1271e-3)},
         'ne60': {'Bilinear':(4.9334e-3, 5.2060e-3, 5.3883e-3), 
                  'V-GECoRe':(2.7944e-2, 2.7674e-2, 3.9585e-2),
                  'RBF'     :(1.9131e-3, 2.0388e-3, 2.5105e-3)},
         'ne120':{'Bilinear':(1.2456e-3, 1.3143e-3, 1.3268e-3), 
                  'V-GECoRe':(1.3723e-2, 1.3647e-2, 1.9760e-2),
                  'RBF'     :(1.5707e-3, 1.7298e-3, 2.7313e-3)} }

ll2cs_rotated = {'ne30': {'Bilinear':(2.1549e-2, 2.2601e-2, 2.1892e-2), 
                          'V-GECoRe':(6.2493e-2, 6.2326e-2, 9.3486e-2),
                          'RBF'     :(3.6434e-3, 3.8798e-3, 4.2614e-3)},
                 'ne60': {'Bilinear':(5.4090e-3, 5.6785e-3, 5.6202e-3), 
                          'V-GECoRe':(2.9708e-2, 2.9676e-2, 4.6719e-2),
                          'RBF'     :(2.0136e-3, 2.1311e-3, 2.5799e-3)},
                 'ne120':{'Bilinear':(1.3489e-3, 1.4181e-3, 1.4154e-3), 
                          'V-GECoRe':(1.4631e-2, 1.4650e-2, 2.3605e-2),
                          'RBF'     :(1.6982e-3, 1.8264e-3, 2.7469e-3)} }

cs2ll = {'ne30': {'Bilinear':(2.7550e-2, 3.0431e-2, 6.4118e-2), 
                  'V-GECoRe':(5.0556e-2, 5.3897e-2, 1.4679e-1),
                  'RBF'     :(3.9749e-3, 4.2167e-3, 6.3305e-3),
                  'Lagrange':(1.2190e-3, 1.3152e-3, 2.7027e-3)},
         'ne60': {'Bilinear':(6.8960e-3, 7.6741e-3, 1.5774e-2), 
                  'V-GECoRe':(2.2858e-2, 2.5448e-2, 1.9461e-1),
                  'RBF'     :(2.0054e-3, 2.1584e-3, 3.2950e-3),
                  'Lagrange':(7.6411e-5, 8.3453e-5, 1.7028e-4)},
         'ne120':{'Bilinear':(1.7329e-3, 1.9275e-3, 4.1561e-3), 
                  'V-GECoRe':(1.1032e-2, 1.3787e-2, 1.9622e-1),
                  'RBF'     :(1.5568e-3, 1.7285e-3, 3.2211e-3),
                  'Lagrange':(4.7996e-6, 5.2399e-6, 1.1219e-5)} }

cs2ll_rotated = {'ne30': {'Bilinear':(2.9501e-2, 3.1873e-2, 7.0071e-2),
                          'V-GECoRe':(5.2149e-2, 5.9707e-2, 1.7977e-1),
                          'RBF'     :(4.9512e-3, 5.2617e-3, 1.0800e-2),
                          'Lagrange':(1.5751e-3, 1.6132e-3, 2.9568e-3)},
                 'ne60': {'Bilinear':(7.4322e-3, 8.0143e-3, 1.6562e-2), 
                          'V-GECoRe':(2.2753e-2, 2.7588e-2, 9.1543e-2),
                          'RBF'     :(2.4327e-3, 2.5862e-3, 5.5198e-3),
                          'Lagrange':(1.0084e-4, 1.0255e-4, 1.8064e-4)},
                 'ne120':{'Bilinear':(1.8548e-3, 2.0030e-3, 4.3244e-3), 
                          'V-GECoRe':(1.0847e-2, 1.3505e-2, 4.8662e-2),
                          'RBF'     :(1.7603e-3, 1.9059e-3, 3.7011e-3),
                          'Lagrange':(6.3101e-6, 6.4270e-6, 1.1773e-5)} }



#-----------------------------------------------------------------------------
# Plot
#-----------------------------------------------------------------------------
direction = 'll2cs'
rotated = True
styles = {'Bilinear':'ko', 'V-GECoRe':'rs', 'RBF':'cD', 'Lagrange':'b^'}


plt.ion()
#fig = plt.figure(figsize=(12,16))
#fig.subplots_adjust()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

exp = '%s_rotated'%direction if rotated else direction

ste_dict = dict()
for method in globals()[exp]['ne30'].keys():
    ste_dict[method] = {'L1':[], 'L2':[], 'Linf':[]}

for ne in [30, 60, 120]:
    for method, (l1,l2,linf) in globals()[exp]['ne%d'%ne].items():
        ste_dict[method]['L1'].append(l1)
        ste_dict[method]['L2'].append(l2)
        ste_dict[method]['Linf'].append(linf)

lines = list()
for method, stes in ste_dict.items():
    line1, = ax.plot(stes['L1'], '%s-'%styles[method], label=method)
    line2, = ax.plot(stes['L2'], '%s--'%styles[method])
    line3, = ax.plot(stes['Linf'], '%s:'%styles[method])
    lines.append(line1)

cs_name = 'Rotated Cubed-sphere' if rotated else 'Cubed-sphere'
if direction == 'll2cs':
    ax.set_title('Latlon -> %s'%cs_name)
    ax.set_ylim([-0.01,0.1]) 
else:
    ax.set_title('%s -> Latlon'%cs_name)
    ax.set_ylim([-0.01,0.08])

ax.set_xlim([-0.2,2.2])
ax.set_xticks([0,1,2])
ax.set_xticklabels([100,50,25])
ax.set_xlabel('Resolution (km)')
ax.set_ylabel('Standard Error')
legend1 = ax.legend(handles=lines, loc='upper right')
ax.add_artist(legend1)

line1, = ax.plot([], 'k-', label='L1')
line2, = ax.plot([], 'k--', label='L2')
line3, = ax.plot([], 'k:', label='Linf')
ax.legend(handles=[line1,line2,line3], loc='upper center')

plt.savefig('%s.png'%exp, dpi=150)
plt.show(True)
