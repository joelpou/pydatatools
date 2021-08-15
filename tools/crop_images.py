import sys
import os
from PIL import Image
import dlib
import numpy as np
import glob
import time

inputDir = str(sys.argv[1])  # input with all images
outputDir = str(sys.argv[2])  # output dir
os.makedirs(outputDir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
offset_t = 150
offset_b = 75
offset_lr = 50
start = time.time()


for f in glob.glob(os.path.join(inputDir, "*.jpg")):
    im = Image.open(f)
    a = np.asarray(im)
    dets = detector(a, 1)

    for i, rect in enumerate(dets):
        left = rect.left() - offset_lr
        top = rect.top() - offset_t
        right = rect.right() + offset_lr
        bottom = rect.bottom() + offset_b
        img_cropped = im.crop((left, top, right, bottom))
        # img_cropped.show()
        basename = os.path.basename(f)
        stringx = basename.rsplit("_", 1)
        if len(dets) > 1:
            fullname = stringx[0] + "_" + str(i) + "_" + stringx[1]
        else:
            fullname = stringx[0] + "_" + stringx[1]
        fullpath = os.path.join(outputDir, fullname)
        print('saving: ' + fullpath)
        img_cropped.save(fullpath)
end = time.time()
print("Done in " + str((end - start) / 60) + " minutes.")

# basename = os.path.basename(f)
# basename = os.path.basename("watchingandersoncooper360andersoncooper36007152021s2021e146slingtvgooglechrome20210715204454_159")
