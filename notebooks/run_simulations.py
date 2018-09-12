import os
import subprocess as sub

filtered = ['uncontrolled.inp', 'uncontrolled_diff.inp']
swmmpath = '/home/mdbartos/Git/USEPA.Stormwater-Management-Model/swmm5'

for fn in os.listdir('../data/inp/'):
    if (not fn in filtered) and ('diff' in fn):
        basename = fn.split('.inp')[0]
        if not '{0}.out'.format(basename) in os.listdir('../data/out'):
            print(basename)
            commands = [swmmpath,
                        '../data/inp/{0}.inp'.format(basename),
                        '../data/rpt/{0}.rpt'.format(basename),
                        '../data/out/{0}.out'.format(basename),
            ]
            sub.run(commands)
