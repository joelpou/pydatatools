# Script to print some info about dataset

import pandas as pd
import sys
import glob

# input folder with all csvs

inputDir = str(sys.argv[1])  # input path where csv's are stored

for file in glob.iglob(inputDir + '**/*.csv', recursive=True):
    print('Dataset: ' + file)
    df = pd.read_csv(file)

    spoofs_cnt = 0
    reals_cnt = 0
    total = df.shape[0]

    print('Total Samples: ' + str(total))

    for i in range(total):
        if df.iloc[i, 0] == 1:
            reals_cnt = reals_cnt + 1
        else:
            spoofs_cnt = spoofs_cnt + 1

    print('Number of reals: ' + str(reals_cnt) + ' (' + str(int(100*reals_cnt/total)) + '%)')
    print('Number of spoofs: ' + str(spoofs_cnt) + ' (' + str(int(100*spoofs_cnt/total)) + '%)')
    print('\n')


