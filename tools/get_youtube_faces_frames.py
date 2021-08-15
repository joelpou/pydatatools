import random
import sys
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import dlib

npz_path = sys.argv[1]
output_imgs = str(sys.argv[2])  # output path to dump all imgs converted
os.makedirs(output_imgs, exist_ok=True)
frames_wanted = 10

detector = dlib.get_frontal_face_detector()


def get_random_numbers(num_wanted, num_range):
    num_list = []
    for num in range(num_wanted):
        rand_num = random.randint(0, num_range)
        while rand_num in num_list:
            rand_num = random.randint(0, num_range)
        num_list.append(rand_num)
    return num_list


# print(os.getcwd())
# os.startfile(r"C:\Users\Tesla\Code\pydatatools\youtube_faces\face_detection_ex")
for fnpz in glob.glob(os.path.join(npz_path + "**/*.npz"), recursive=True):
    print("Processing file: {}".format(fnpz))
    npz = np.load(fnpz)
    for file in npz:
        if "colorImages" in file:
            # extract npy
            npy = npz[file]

            # extract first frame from video
            fFrame = npy[:, :, :, 0]

            # detect faces
            dets = detector(fFrame, 1)

            if len(dets) == 1:
                fDetWidth = dets[0].right() - dets[0].left()
                print('rect width: ' + str(fDetWidth))
                if fDetWidth >= 160:
                    # get indexes to select random frames from vid
                    frame_index_list = get_random_numbers(frames_wanted, npy.shape[3] - 1)
                    for i in frame_index_list:
                        frame = npy[:, :, :, i]
                        # plt.imshow(frame)
                        # plt.show()
                        img = Image.fromarray(frame)
                        # print(df.shape)
                        outputImg = output_imgs + os.path.splitext(os.path.basename(fnpz))[0] + '_' + str(i) + '.jpg'
                        print('writing ' + outputImg)
                        img.save(outputImg)
                    print('\n')
                else:
                    print('detection rectangle width too small! skipping...')
            else:
                print('more than one face detected or no faces detected! skipping...')
    print('\n')

# if "landmark2d" in npy:
#     # get landmarks
#     landmarks = np.zeros((npy.shape[0], npy.shape[1]))
#     for i in frame_index_list:
#         landmarks = npy[:, :, i]
#         # plt.imshow(frame)
#         # plt.show()
#         img = Image.fromarray(frame)
#         # print(df.shape)
#         outputImg = output_imgs + os.path.splitext(os.path.basename(fnpz))[0] + '_' + str(i) + '.jpg'
#         print('writing ' + outputImg)
#         img.save(outputImg)
#     print('\n')
# print(type(npy))
# # reference
#
# shape = np.zeros((npy.shape[0], npy.shape[1]))
# # for loop per frame npy[landmark_index, point (x,y), frame_index]
# for frame in range(npy.shape[2]):
#     for landmark in range(npy.shape[0]):
#         for point in range(npy.shape[1]):
#             shape[landmark, point] = npy[landmark, point, frame]

# df = pd.DataFrame()
# for i, part in enumerate(shape.parts()):
#     ls = pd.Series([i, int(part.x), int(part.y)])
#     ld = pd.DataFrame([ls])
#     df = pd.concat([df, ld])  # concat each landmark as new row on dataframe
#
# ls = pd.Series([d.left(), d.top(), d.right(), d.bottom()])
# ld = pd.DataFrame([ls])
# df = pd.concat([df, ld])  # add rect points to last row
#
# print(df.shape)
# outputCsv = output_csvs + os.path.splitext(os.path.basename(f))[0] + '.csv'
# print('writing ' + outputCsv)
# df.to_csv(outputCsv, header=False, index=False)
# print('\n')
