#------------------------------------------------------------------------------
# filename  : load_yaml.py
# author    : Ki-Hwan Kim  (kh.kim@kiaps.org)
# affilation: KIAPS (Korea Institute of Atmospheric Prediction Systems)
# update    : 2016.9.12    Start
#             2016.9.20    Move the select section to root
#
# Wrap the PyYAML
#   - bug fix for a exponential number type
#   - replace substitution variables
#------------------------------------------------------------------------------

import numpy
import re
import yaml




def replace_braket(serial_dict, key):
    cb = re.compile('<([\w]+)>')     # find pattern <some>
    val = serial_dict[key]
    #print('k,v: ', key, val)

    if type(val) == str and len(cb.findall(val)) > 0:
        ss = val

        for k2 in cb.findall(val):
            v2 = serial_dict[k2]
            #print('k2,v2: ', k2, v2)

            if type(v2) == str and cb.match(v2):
                v2 = replace_braket(serial_dict, k2)
                #print('\tv2: ', v2)

            ss = ss.replace('<{}>'.format(k2), '({})'.format(v2))

        #print('\tss: ', ss)
        #print('\teval(ss): ', eval(ss))
        return eval(ss)

    elif type(val) == str and re.match('([-+]?[0-9.]*[eE][-+]?[0-9]+)', val):
        return float(val)

    else:
        return val




def replace_braket_tree(src_dict, dst_dict, serial_dict, key):
    val = src_dict[key]
    #print('k,type(v): ', key, type(val))

    if type(val) == dict:
        dst_dict[key] = dict()

        for k in val.keys():
            replace_braket_tree(val, dst_dict[key], serial_dict, k)

    else:
        dst_dict[key] = replace_braket(serial_dict, key)




def serialize_dict(src_dict, serial_dict, key):
    val = src_dict[key]

    if type(val) == dict:
        for k in val.keys():
            serialize_dict(val, serial_dict, k)

    else:
        serial_dict[key] = val




def load_yaml_dict(fpath, select=[]):
    with open(fpath, 'r') as f: src_dict = yaml.load(f)

    serial_dict = dict()
    dst_dict = dict()

    #
    # Filter 
    #
    if len(select) > 0:
        pattern = ''.join(['{}([\d]+)'.format(k) for k,v in select])
        sname = ''.join(['{}{}'.format(k,v) for k,v in select])
        cb = re.compile(pattern)
        #print(sname, pattern)

    #
    # Serialize
    #
    for key in src_dict.keys():
        if len(select) > 0 and len(cb.findall(key)) == 1 and key != sname: continue
        serialize_dict(src_dict, serial_dict, key)
    #print(serial_dict)

    #
    # Replace substitution variables
    #
    for key in src_dict.keys():
        if len(select) > 0 and len(cb.findall(key)) == 1 and key != sname: continue
        replace_braket_tree(src_dict, dst_dict, serial_dict, key)
    #print(dst_dict)

    #
    # Move the select section to root
    #
    if len(select) > 0:
        dst_dict.update( dst_dict.pop(sname) )

    return dst_dict
