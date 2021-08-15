import sys
import os
import dlib
import glob
import statistics as stats

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
faces = []
rect_widths = []
rect_heights = []
file = open("rejected_images.txt", "w")
file.write("Rejected images: \n")
file.close()

for f in glob.glob(os.path.join(faces_folder_path, "**/*.png"), recursive=True):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    det = detector(img, 1)
    num_faces = len(det)
    if num_faces > 1 or num_faces == 0:
        print('')
        print("detector error in '{}', skipping image...".format(f))
        file = open("rejected_images.txt", "a")
        file.write('\n')
        file.write(f)
        file.close()
        print('')
        continue
    else:
        face = {}
        for rect in det:
            print(rect)
            face['img_name'] = os.path.basename(f)
            face['rect_width'] = rect.width()
            face['rect_height'] = rect.height()
            rect_widths.append(rect.width())
            rect_heights.append(rect.height())
        faces.append(face)

print(faces)
# mean, std, max, min
print('')
print('mean: ')
print('width: ' + str(round(stats.mean(rect_widths), 3)))
print('height: ' + str(round(stats.mean(rect_heights), 3)))
print('')
print('stdev: ')
print('width: ' + str(round(stats.stdev(rect_widths), 3)))
print('height: ' + str(round(stats.stdev(rect_heights), 3)))
print('')
print('max: ')
print('width: ' + str(max(rect_widths)))
print('height: ' + str(max(rect_heights)))
print('')
print('min: ')
print('width: ' + str(min(rect_widths)))
print('height: ' + str(min(rect_heights)))


