#!/usr/bin/env python

import pickle

data1 = {'a': [1, 2.0, 3, 4+6j], \
        'b': ('string', u'Unicode string'), \
        'c': None, \
        'd': slice(2, -1)}

selfref_list = [1, 2, 3]
selfref_list.append(selfref_list)

output = open('pickle_data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(data1, output)

# Pickle the list using the highest protocol available.
pickle.dump(selfref_list, output, -1)

output.close()
