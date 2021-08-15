import os
import sys
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

root = os.getcwd()  # root path
input = str(sys.argv[1])  # path where all videos are stored
output = str(sys.argv[2])  # path to dump trimmed videos
start = str(sys.argv[3])  # start time
end = str(sys.argv[4])  # end time
inputDir = os.path.join(root, input)
outputDir = os.path.join(root, output)
os.makedirs(inputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)

for video in glob.glob(inputDir + '**/*', recursive=True):
    if os.path.isfile(video):
        print('trimming: ' + video)
        ofilename = os.path.basename(os.path.splitext(video)[0]) + '_trim' + os.path.splitext(video)[1]
        ofilepath = os.path.join(outputDir, ofilename)
        ffmpeg_extract_subclip(video, int(start), int(end), targetname=ofilepath)
