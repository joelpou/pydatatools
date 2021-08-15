import face_alignment
from skimage import io
import pandas as pd
import sys
import os
import glob
import math
import numpy as np
import dlib

faces_folder_path = sys.argv[1]
output_csvs = str(sys.argv[2])  # output path to dump all csv's converted
output_imgs = str(sys.argv[3])  # output path to dump all imgs used for landmark extraction (not skipped)
os.makedirs(output_csvs, exist_ok=True)
os.makedirs(output_imgs, exist_ok=True)

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold": 0.8
}

# Run the 2D/3D face alignment (can be run with CUDA).
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)

left_eye_indexes = [36, 37, 38, 39, 40, 41]
right_eye_indexes = [42, 43, 44, 45, 46, 47]


def get_midpoint(p1, p2):
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    return x, y


def get_euclidean_distance(leftx, lefty, rightx, righty):
    return math.sqrt(float((leftx - rightx) ** 2 + (lefty - righty) ** 2))


def get_eye_center_point(eye_points, facial_landmarks):
    center_top = get_midpoint(facial_landmarks[eye_points[1]], facial_landmarks[eye_points[2]])
    center_bot = get_midpoint(facial_landmarks[eye_points[5]], facial_landmarks[eye_points[4]])
    return get_midpoint(center_top, center_bot)


def get_iod_ok(left_eye_points, right_eye_points, facial_landmarks):
    left_center_point = get_eye_center_point(left_eye_points, facial_landmarks)
    right_center_point = get_eye_center_point(right_eye_points, facial_landmarks)
    dist = get_euclidean_distance(left_center_point[0], left_center_point[1], right_center_point[0],
                                  right_center_point[1])
    return dist > 50


# input_img = io.imread("../../pydatatools/cew/closed_eye_1819.jpg_face_1.jpg")
# preds = fa.get_landmarks(input_img)
# print(preds)

log = open("skippedImages.txt", "a")
detector = dlib.get_frontal_face_detector()

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    im = io.imread(f)
    preds = fa.get_landmarks_from_image(im, return_bboxes=True)
    dets = detector(im, 1)

    if preds is None or len(dets) == 0:
        print('skipping...\n')
        log.write(f + "\n")
        continue

    d = dets[0]
    marks = preds[0][0]
    # boxes = np.delete(preds[1][0], -1).astype(int) # Use fan detection rect
    boxes = [d.left(), d.top(), d.right(), d.bottom()]  # Use dlib detection rect
    # check if interocular distance from points is greater than 50 pixels
    iod_ok = get_iod_ok(left_eye_indexes, right_eye_indexes, preds[0][0])

    if not iod_ok:
        print('skipping...\n')
        log.write(f + "\n")
        continue

    df = pd.DataFrame(marks, dtype=int)
    ls = pd.Series([boxes[0], boxes[1], boxes[2], boxes[3]])
    ld = pd.DataFrame([ls])
    df = pd.concat([df, ld], ignore_index=True)  # add rect points to last row
    outputCsv = output_csvs + os.path.splitext(os.path.basename(f))[0] + '.csv'
    print('writing ' + outputCsv)
    df.to_csv(outputCsv, header=False, index=True)
    outputImg = output_imgs + os.path.basename(f)
    io.imsave(outputImg, im)
    print('\n')

print('done.')
log.close()
