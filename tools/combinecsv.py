import pandas as pd
import os
import sys
import glob

# input folder with all csvs
# output csv with all vectors

inputDir = str(sys.argv[1])  # input path where csv's are stored
outputFile = str(sys.argv[2])
sampleNumber = str(sys.argv[3])

outputDir = os.path.join(inputDir, outputFile)

samples = int(sampleNumber)

ext = 'csv'
files = [i for i in glob.glob(inputDir + '*.{}'.format(ext))]

# fyi, never try concatenating many csv's into a single csv using pandas.concat shit function
with open(outputDir, 'w') as outfile:
    for file in files:
        cnt = 0
        print(file)
        with open(file, 'r') as infile:
            for line in infile:
                if 0 < samples == cnt:
                    break
                else:
                    outfile.write(line + '\n')
                    if 0 < samples:
                        cnt = cnt + 1

print(pd.read_csv(outputDir))
