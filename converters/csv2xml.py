import csv
import os
import sys
import glob

input_csv_folder = str(sys.argv[1])
input_image_folder = str(sys.argv[2])
output_file = str(sys.argv[3])


def convert_row(row):
    return """\t\t\t\t<part name='{:0>2d}' x='{:.0f}' y='{:.0f}'/>""".format(int(row[0]), float(row[1]), float(row[2]))


def process_csv(input_file):
    # input_filename='closed_eye_0002.jpg_face_2.csv'
    f = open(input_file)
    csv_f = csv.reader(f)
    landmarks = []
    #
    for row in csv_f:
        landmarks.append(row)
    f.close()
    # print (data[:])
    input_filenamex = input_file.rsplit(".csv")
    input_image_filename = os.path.basename(input_filenamex[0] + ".jpg")
    rect = landmarks[68]
    # load image and get dims
    # image = Image.open(input_image_folder + input_image_filename)
    # return (input_image_filename, str(image.width), str(image.height), landmarks)
    return input_image_filename, rect, landmarks[:-1]


# print ('\n'.join([convert_row(row) for row in data[:]]))

print('start')

with open(output_file, 'w') as f:
    f.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
    f.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
    f.write("<dataset>\n")
    f.write("\t<name>FW landmarks</name>\n")
    f.write(
        "\t<comment>These are images from the FW dataset. The face landmarks are from FAN landmarking model.</comment>\n")
    f.write("\t<images>\n")
    #
    for csvFile in glob.glob(os.path.join(input_csv_folder, '*.csv')):
        print('now processing file: {0}'.format(csvFile))
        # filenamex = filename.rsplit('/', 1)
        input_image_filename, box, data = process_csv(csvFile)
        width = int(float(box[3]))-int(box[1])
        height = int(float(box[4]))-int(box[2])
        # imgPath = ""
        # for img in glob.glob(os.path.join(input_image_folder, '*.jpg')):
        #     if input_image_filename in img:
        #         imgPath = img
        # assert imgPath
        f.write("\t\t<image file='" + input_image_filename + "'>\n")
        f.write("\t\t\t<box top='" + box[2] + "' left='" + box[1] + "' width='" + str(width) + "' height='" + str(height) + "'>\n")
        f.write('\n'.join([convert_row(row) for row in data[:]]))
        f.write("\n\t\t\t</box>\n")
        f.write("\t\t</image>\n")
    #
    f.write("\t</images>\n")
    f.write("</dataset>\n")

print('done')
