import sys
import os
from PIL import Image
import dlib
import numpy as np
import glob
import time
from concurrent.futures import ProcessPoolExecutor

inputDir = str(sys.argv[1])  # input with all images
outputDir = str(sys.argv[2])  # output dir
os.makedirs(outputDir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
offset_t = 150
offset_b = 75
offset_lr = 50


def crop_images(dets, imp, imf):
    for i, rect in enumerate(dets):
        left = rect.left() - offset_lr
        top = rect.top() - offset_t
        right = rect.right() + offset_lr
        bottom = rect.bottom() + offset_b
        img_cropped = imp.crop((left, top, right, bottom))
        basename = os.path.basename(imf)
        stringx = basename.rsplit("_", 1)
        if len(dets) > 1:
            fullname = stringx[0] + "_" + str(i) + "_" + stringx[1]
        else:
            fullname = stringx[0] + "_" + stringx[1]
        fullpath = os.path.join(outputDir, fullname)
        print('saving: ' + fullpath)
        img_cropped.save(fullpath)


def get_rects(img_file):
    print(img_file)
    img_pil = Image.open(img_file)
    a = np.asarray(img_pil)
    dets = detector(a, 1)
    return dets, img_pil, img_file


def main():
    start = time.time()
    with ProcessPoolExecutor() as executor:
        image_files = [f for f in glob.glob(inputDir + "/*.jpg")]
        for detections, im, file in executor.map(get_rects, image_files):
            crop_images(detections, im, file)
    end = time.time()
    print("Done in " + str((end - start) / 60) + " minutes.")


if __name__ == '__main__':
    main()
