import os
import sys
import glob
from zipfile import ZipFile

# How to run on terminal: python3 file_zip_extract.py zips/ mp4s/ mp4
root = os.getcwd()
input = str(sys.argv[1])  # path where all zips are stored
output = str(sys.argv[2])  # path to dump all videos
outputFormat = str(sys.argv[3])  # output video format
inputDir = os.path.join(root, input)
outputDir = os.path.join(root, output)
os.makedirs(inputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)
for zfile in glob.glob(inputDir + '**/*.zip', recursive=True):
    print('zip: ' + zfile)
    if os.path.isfile(zfile):
        with ZipFile(zfile, 'r') as zipObj:
            filelist = zipObj.namelist()
            for filename in filelist:
                if filename.endswith('.' + outputFormat):
                    print('extracting: ' + filename)
                    zipObj.extract(filename, outputDir)

