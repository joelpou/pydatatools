import sys
import os
import glob
from PIL import Image

inputDir = sys.argv[1]
outputDir = str(sys.argv[2])  # output path to dump all images converted
ext2convert = str(sys.argv[3])  # extension to convert image to
os.makedirs(outputDir, exist_ok=True)

for f in glob.iglob(inputDir + '*'):
    print("Converting image: {}".format(f) + " to " + ext2convert)
    img = Image.open(f)
    file_out = outputDir + os.path.splitext(os.path.basename(f))[0] + '.' + ext2convert
    img.save(file_out)

