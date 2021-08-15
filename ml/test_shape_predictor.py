import os
import sys
import glob
import dlib
import pandas as pd
import numpy as np
import statistics as stats

testing_faces_path = str(sys.argv[1])
testing_annotations_path = str(sys.argv[2])
sp_path = str(sys.argv[3])

predictor = dlib.shape_predictor(sp_path)
detector = dlib.get_frontal_face_detector()
win = dlib.image_window()


def interocular_distance(det):
    r = dlib.vector(2)
    l = dlib.vector(2)
    cnt = 0
    for j in range(0, 6):
        l[0] += det.part(j).x
        l[1] += det.part(j).y
        cnt = cnt + 1
    l[0] = l[0]/cnt
    l[1] = l[1]/cnt
    cnt = 0
    for j in range(6, 12):
        r[0] += det.part(j).x
        r[1] += det.part(j).y
        cnt = cnt + 1
    r[0] = r[0]/cnt
    r[1] = r[1]/cnt

    diff = dlib.point(l)-dlib.point(r)
    return dlib.length(diff)


def get_eye_lines(eye_parts):
    line_list = []
    last_point = dlib.point(eye_parts[0])
    first_point = last_point
    for p in eye_parts:
        curr_point = dlib.point(p)
        line = dlib.line(curr_point, last_point)
        line_list.append(line)
        last_point = curr_point
    line_list.append(dlib.line(last_point, first_point))  # close eye loop
    return line_list


for s in glob.glob(os.path.join(testing_annotations_path, "*.csv")):
    shapenamex = os.path.basename(s).rsplit(".csv")[0]
    for f in glob.glob(os.path.join(testing_faces_path, "*.jpg")):
        if shapenamex in f:
            print("Processing file: {}".format(f))
            img = dlib.load_rgb_image(f)

            win.clear_overlay()
            win.set_image(img)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                # Get the landmarks/parts for the face in box d.
                inferred_shape = predictor(img, d)

                # id = interocular_distance(inferred_shape)
                # print("Part 0: {}, Part 1: {} ...".format(inferred_shape.part(0),
                #                                           inferred_shape.part(1)))
                # Draw the face landmarks on the screen.
                # win.add_overlay(inferred_shape, dlib.rgb_pixel(0, 255, 0))

                # get landmark data from annotations to compare with inferred
                df = pd.read_csv(s, header=None)
                box = df.values[-1]
                rect = dlib.rectangle(int(box[1]), int(box[2]), int(box[3]), int(box[4]))

                annotated_parts = df.values[:-1]  # get numpy array from dataframe minus last row which contains rect
                annotated_parts = np.delete(annotated_parts, 0, 1)[:, :2]  # slice off first and last two columns
                point_list = []
                for p in annotated_parts:
                    point = dlib.point(p)  # cast all parts to dlib points
                    point_list.append(point)
                annotated_shape = dlib.full_object_detection(rect, point_list)

                if len(annotated_parts) < 68:
                    # draw inferred eyes
                    inferred_parts = np.zeros((12, 2))
                    for i, part in enumerate(inferred_shape.parts()):
                        inferred_parts[i, 0] = part.x
                        inferred_parts[i, 1] = part.y
                    left_eye_lines = get_eye_lines(inferred_parts[:6])
                    right_eye_lines = get_eye_lines(inferred_parts[6:])
                    eyes_list = left_eye_lines + right_eye_lines
                    win.add_overlay(eyes_list, dlib.rgb_pixel(0, 255, 0))

                    # draw annotated eyes
                    left_eye_lines = get_eye_lines(annotated_parts[:6])
                    right_eye_lines = get_eye_lines(annotated_parts[6:])
                    eyes_list = left_eye_lines + right_eye_lines
                    win.add_overlay(eyes_list, dlib.rgb_pixel(0, 0, 255))

                    # get mean error
                    score = []
                    for k, part in enumerate(inferred_shape.parts()):
                        diff = part - annotated_shape.part(k)
                        score.append(dlib.length(diff))
                    mean = stats.mean(score)

                else:
                    annotated_shape = dlib.full_object_detection(rect, point_list)
                    win.add_overlay(annotated_shape, dlib.rgb_pixel(0, 0, 255))

            win.add_overlay(dets, dlib.rgb_pixel(255, 0, 0))
            dlib.hit_enter_to_continue()
