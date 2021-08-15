import math
import sys
import os
import dlib
import glob
import pandas as pd

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
output_imgs = str(sys.argv[3])  # output path to dump all images filtered
output_csvs = str(sys.argv[4])  # output path to dump all csv's converted
os.makedirs(output_imgs, exist_ok=True)
os.makedirs(output_csvs, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

left_eye_indexes = [36, 37, 38, 39, 40, 41]
right_eye_indexes = [42, 43, 44, 45, 46, 47]

# win = dlib.image_window()

def get_midpoint(p1, p2):
    x = (p1.x + p2.x) / 2
    y = (p1.y + p2.y) / 2
    return dlib.dpoint(x, y)


def get_euclidean_distance(leftx, lefty, rightx, righty):
    return math.sqrt(float((leftx - rightx) ** 2 + (lefty - righty) ** 2))


def get_eye_center_point(eye_points, facial_landmarks):
    center_top = get_midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bot = get_midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    return get_midpoint(center_top, center_bot)


def get_iod_ok(left_eye_points, right_eye_points, facial_landmarks):
    left_center_point = get_eye_center_point(left_eye_points, facial_landmarks)
    right_center_point = get_eye_center_point(right_eye_points, facial_landmarks)
    dist = get_euclidean_distance(left_center_point.x, left_center_point.y, right_center_point.x, right_center_point.y)
    return dist > 50


for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # win.clear_overlay()
    # win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) != 1:
        print('skipping...\n')
        continue

    dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))

        # check if interocular distance from points is greater than 50 pixels
        iod_ok = get_iod_ok(left_eye_indexes, right_eye_indexes, shape)

        # if you want to curate dataset
        if shape.num_parts > 0 and iod_ok:
            dlib.save_image(img, output_imgs + os.path.basename(f))
        else:
            print('skipping...\n')
            continue

        df = pd.DataFrame()
        for i, part in enumerate(shape.parts()):
            ls = pd.Series([i, int(part.x), int(part.y)])
            ld = pd.DataFrame([ls])
            df = pd.concat([df, ld])  # concat each landmark as new row on dataframe

        ls = pd.Series([d.left(), d.top(), d.right(), d.bottom()])
        ld = pd.DataFrame([ls])
        df = pd.concat([df, ld])  # add rect points to last row

        print(df.shape)
        outputCsv = output_csvs + os.path.splitext(os.path.basename(f))[0] + '.csv'
        print('writing ' + outputCsv)
        df.to_csv(outputCsv, header=False, index=False)
        print('\n')

        # Draw the face landmarks on the screen.
    # win.add_overlay(shape)
    #
    # win.add_overlay(dets)
    # dlib.hit_enter_to_continue()
