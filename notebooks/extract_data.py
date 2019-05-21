import os
import numpy as np
import pandas as pd
from swmmtoolbox import swmmtoolbox

output_dir = '../data/out'

data = {}

for fn in os.listdir(output_dir):
    if fn.endswith('.out'):
        basename = fn.split('.out')[0]
        outfall = swmmtoolbox.extract('../data/out/{0}'.format(fn), 'system,Flow_leaving_outfalls,11')
        data[basename] = outfall

for fn in data:
    data[fn].columns = [fn]

df = pd.concat(data.values(), axis=1).sort_index(axis=1)
df.to_csv('../data/all_outflows.csv')
