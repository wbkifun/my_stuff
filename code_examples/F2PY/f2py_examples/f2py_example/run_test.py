import sys
import os
import pylab as P
subfolders = ['basic', 'intermediate', 'advanced']
# append path to the folder
sys.path.extend(subfolders)

# go through all the folders, make the python module and run the
# program
f2py_module_list = []
run_module_list = []
for i,folder in enumerate(subfolders):
    # complie the model
    print 'compling xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    os.chdir(folder)
    try:
        os.remove('simple_ice_model_%s.so' % folder) # first remove the old one
    except OSError:
        pass
    os.system('f2py -c --fcompiler=gnu95 simple_ice_model_%s.pyf ../simple_ice_model_modules.f90 simple_ice_model_%s.f90' %tuple([folder for i in range(2)]))
    os.chdir('..')
    # import the module
    f2py_module_list.append(__import__('simple_ice_model_%s' % folder))
    run_module_list.append(__import__('run_%s' % folder))
    print 'running model xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    run_module_list[-1].run()

P.show()



