import sys
import os
import dlib
import glob

# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n")
#     exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
outputDir = str(sys.argv[3])  # output path to dump all chips
os.makedirs(outputDir, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

for f in glob.glob(os.path.join(faces_folder_path, "*.bmp")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(f))
        exit()

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(predictor(img, detection))

    # window = dlib.image_window()

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=250, padding=0.25)
    for image in images:
        # window.set_image(image)
        output = outputDir + os.path.splitext(os.path.basename(f))[0] + '.bmp'
        print('writing ' + output)
        dlib.save_image(image, output)
        # dlib.hit_enter_to_continue()

    # # It is also possible to get a single chip
    # image = dlib.get_face_chip(img, faces[0])
    # window.set_image(image)
    # dlib.hit_enter_to_continue()
