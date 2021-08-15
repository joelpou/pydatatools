import sys
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot


def load_images(folder, info):
    limages = []
    for filename in os.listdir(folder):
        if filename.startswith(".") is False:
            file = os.path.join(folder, filename)
            image = Image.open(file)
            if info:
                print(file)
                print(image.format)
                print(image.mode)
                print(image.size)
                print(image.palette)
                print('\n')
            if image is not None:
                limages.append(image)
    return limages


inputDir = str(sys.argv[1])  # input with all images to convert to iqm
outputDir = str(sys.argv[2])  # output dir
name = str(sys.argv[3])  # output file name
os.makedirs(outputDir, exist_ok=True)

# Example params:
# ../datasets/other/spoof-tests/spoof/ ../datasets/other/spoof-tests/spoof-ready/ spoof

imgs = load_images(inputDir, False)

for i, img in enumerate(imgs):
    if i < 10:
        subs = '_0'
    else:
        subs = '_'
    fullname = name + subs + str(i) + '.jpg'
    print('saving: ' + fullname)
    img.save(os.path.join(outputDir, fullname))
