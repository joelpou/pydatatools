# Script to standarize Replay Mobile Dataset Feature Vectors

import pandas as pd
import os
import sys
import glob

# input folder with all csvs
# output csv with all vectors

inputDir = str(sys.argv[1])  # input path where csv's are stored
outputDir = str(sys.argv[2])
os.makedirs(outputDir, exist_ok=True)

# sanitize csv's (remove first column and row)
# add first column with label and second column with file name
for file in glob.iglob(inputDir + '*.csv'):
    print(file)
    df = pd.read_csv(file, index_col=0, skiprows=1)
    print(df)
    name = os.path.splitext(os.path.basename(file))[0] # get file name w/o ext
    if "attack" in name:
        label = -1
    else:
        label = +1

    df.insert(0, None, label, True)
    df.insert(1, None, name, True)
    csvOut = outputDir + os.path.basename(file)
    df.to_csv(csvOut, index=False, header=0)
    dfo = pd.read_csv(csvOut, index_col=0)
    print(dfo)
