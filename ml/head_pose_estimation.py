import face_alignment
from skimage import io
import pandas as pd
import dlib
import cv2
import sys
import os
import glob
import math
import numpy as np

faces_folder_path = sys.argv[1]
output_csvs = str(sys.argv[2])  # output path to dump all csv's converted
os.makedirs(output_csvs, exist_ok=True)

detector = dlib.get_frontal_face_detector()

# Optionally set detector and some additional detector parameters
face_detector = 'sfd'
face_detector_kwargs = {
    "filter_threshold": 0.8
}

# Run the 3D face alignment on a test image, without CUDA.
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


def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    """
    Draw a 3D anotation box on the face for head pose estimation

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix
    rear_size : int, optional
        Size of rear box. The default is 300.
    rear_depth : int, optional
        The default is 0.
    front_size : int, optional
        Size of front box. The default is 500.
    front_depth : int, optional
        Front depth. The default is 400.
    color : tuple, optional
        The color with which to draw annotation box. The default is (255, 255, 0).
    line_width : int, optional
        line width of lines drawn. The default is 2.

    Returns
    -------
    None.

    """

    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    """
    Get the points to estimate head pose sideways

    Parameters
    ----------
    img : np.unit8
        Original Image.
    rotation_vector : Array of float64
        Rotation Vector obtained from cv2.solvePnP
    translation_vector : Array of float64
        Translation Vector obtained from cv2.solvePnP
    camera_matrix : Array of float64
        The camera matrix

    Returns
    -------
    (x, y) : tuple
        Coordinates of line to estimate head pose

    """
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)


def draw_pose_estimation(img, marks, model_points, camera_matrix, font):
    image_points = np.array([
        marks[30],  # Nose tip
        marks[8],  # Chin
        marks[36],  # Left eye left corner
        marks[45],  # Right eye right corne
        marks[48],  # Left Mouth corner
        marks[54]  # Right mouth corner
    ], dtype="double")
    image_points = np.ascontiguousarray(image_points[:, :2]).reshape((image_points.shape[0], 1, 2))
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        center = (int(p[0][0]), int(p[0][1]))
        cv2.circle(img, center, 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0][0]), int(image_points[0][0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

    cv2.line(img, p1, p2, (0, 255, 255), 2)
    cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
    # for (x, y) in marks:
    #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
    # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
    try:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        ang1 = int(math.degrees(math.atan(m)))
    except:
        ang1 = 90

    try:
        m = (x2[1] - x1[1]) / (x2[0] - x1[0])
        ang2 = int(math.degrees(math.atan(-1 / m)))
    except:
        ang2 = 90

        # print('div by zero error')
    if ang1 >= 48:
        print('Head down')
        # cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
    elif ang1 <= -48:
        print('Head up')
        # cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)
    if ang2 >= 48:
        print('Head right')
        # cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
    elif ang2 <= -48:
        print('Head left')
        # cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

    # cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
    # cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)

    print('Angle Pitch : ' + str(ang1))
    print('Angle Yaw: ' + str(ang2))

    draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)
    cv2.imshow('frame', img)
    cv2.waitKey(1)


# input_img = io.imread("../../pydatatools/cew/closed_eye_1819.jpg_face_1.jpg")
# preds = fa.get_landmarks(input_img)
# print(preds)

log = open("skippedImages.txt", "a")

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    preds = fa.get_landmarks_from_image(img, return_bboxes=True)

    if preds is None:
        print('skipping...\n')
        log.write(f + "\n")
        continue

    # img = dlib.load_rgb_image(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print('dlib dets: ' + str(len(dets)))

    # faces = dlib.full_object_detections()
    # for detection in dets:
    #     faces.append(predictor(img, detection))

    # check if interocular distance from points is greater than 50 pixels
    iod_ok = get_iod_ok(left_eye_indexes, right_eye_indexes, preds[0][0])

    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corne
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    if len(preds[1]) > 0:
        p1 = (int(preds[1][0][0]), int(preds[1][0][1]))
        p2 = (int(preds[1][0][2]), int(preds[1][0][3]))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 3)

    if len(dets) > 0:
        pd1 = (dets[0].left(), dets[0].top())
        pd2 = (dets[0].right(), dets[0].bottom())
        cv2.rectangle(img, pd1, pd2, (0, 255, 0), 3)

    widthRatio = (preds[1][0][2] - preds[1][0][0])/size[1]
    print(widthRatio)

    draw_pose_estimation(img, preds[0][0], model_points, camera_matrix, font)

    print('\n')

    # if iod_ok:
    #     df = pd.DataFrame(preds[0][0])
    #     print(df.shape)
    #     outputCsv = output_csvs + os.path.splitext(os.path.basename(f))[0] + '.csv'
    #     print('writing ' + outputCsv)
    #     df.to_csv(outputCsv, header=False, index=True)
    #     print('\n')
    # else:
    #     print('skipping...\n')
    #     log.write(f + "\n")
    #     continue
log.close()
