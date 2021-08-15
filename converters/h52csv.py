import pandas as pd
import numpy as np
import h5py
import os
import sys
import glob

dataset = 'bobiqm'
# root = 'datasets/replay/'
# inputDir = os.path.join(root, 'data-hdf5/')
# outputDir = os.path.join(root, 'data-csv-train/')

inputDir = str(sys.argv[1])  # input path where hdf5 are stored
os.makedirs(inputDir, exist_ok=True)
outputDir = str(sys.argv[2])  # output path to dump all csv's converted
os.makedirs(outputDir, exist_ok=True)

for file in glob.iglob(inputDir + '**/*', recursive=True):
    if os.path.isfile(file):
        with h5py.File(file, 'r') as f:
            df = pd.DataFrame(np.array(f[dataset]))
            outputCsv = outputDir + os.path.splitext(os.path.basename(file))[0] + '.csv'
            print('writing ' + outputCsv)
            df.to_csv(outputCsv)
