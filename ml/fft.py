import sys
import os
import dlib
import glob
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

predictor_path = sys.argv[1]
image = sys.argv[2]

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

img = np.array(Image.open(image).convert('L'))
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()
#
# img = dlib.load_rgb_image(f)
#
# # Ask the detector to find the bounding boxes of each face. The 1 in the
# # second argument indicates that we should upsample the image 1 time. This
# # will make everything bigger and allow us to detect more faces.
# dets = detector(img, 1)
#
# num_faces = len(dets)
# if num_faces == 0:
#     print("Sorry, there were no faces found in '{}'".format(f))
#     exit()
#
# # Find the 5 face landmarks we need to do the alignment.
# faces = dlib.full_object_detections()
# for detection in dets:
#     faces.append(predictor(img, detection))
#
# window = dlib.image_window()
#
# # Get the aligned face images
# # Optionally:
# # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
# images = dlib.get_face_chips(img, faces, size=280, padding=0.25)
# for image in images:
#     window.set_image(image)
#     # dlib.hit_enter_to_continue()



