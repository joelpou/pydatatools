import os
import sys
import glob
import dlib

faces_folder = str(sys.argv[1])
output_sp = str(sys.argv[2])
# testing_faces_path = str(sys.argv[3])


options = dlib.shape_predictor_training_options()
# In particular, setting the oversampling
# to a high amount (300) effectively boosts the training set size, so
# that helps this example.
options.oversampling_amount = 1
# I'm also reducing the capacity of the model by explicitly increasing
# the regularization (making nu smaller) and by using trees with
# smaller depths.
options.nu = 0.1
options.tree_depth = 4
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.feature_pool_size = 400
options.num_threads = 5
options.be_verbose = True

# dlib.train_shape_predictor() does the actual training.  It will save the
# final predictor to predictor.dat.  The input is an XML file that lists the
# images in the training dataset and also contains the positions of the face
# parts.
training_xml_path = os.path.join(faces_folder, "train-annotations-eyes.xml")
dlib.train_shape_predictor(training_xml_path, output_sp, options)

# Now that we have a model we can test it.  dlib.test_shape_predictor()
# measures the average distance between a face landmark output by the
# shape_predictor and where it should be according to the truth data.
print("\nTraining accuracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, output_sp)))

testing_xml_path = os.path.join(faces_folder, "test-annotations-eyes.xml")
print("Testing accuracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, output_sp)))

# Now let's use it as you would in a normal application.  First we will load it
# from disk. We also need to load a face detector to provide the initial
# estimate of the facial location.
# predictor = dlib.shape_predictor(output_sp)
# detector = dlib.get_frontal_face_detector()
#
# # Now let's run the detector and shape_predictor over the images in the faces
# # folder and display the results.
# print("Showing detections and predictions on the images in the faces folder...")
# win = dlib.image_window()
# for f in glob.glob(os.path.join(testing_faces_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(f)
#
#     win.clear_overlay()
#     win.set_image(img)
#
#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
#     dets = detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # Get the landmarks/parts for the face in box d.
#         shape = predictor(img, d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
#                                                   shape.part(1)))
#         # Draw the face landmarks on the screen.
#         win.add_overlay(shape)
#
#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()