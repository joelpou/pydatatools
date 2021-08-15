import ffmpeg
import os
import sys
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

root = os.getcwd()  # root path
input = str(sys.argv[1])  # path where all videos are stored
output = str(sys.argv[2])  # path to dump video subfolders with frames
inputDir = os.path.join(root, input)
outputDir = os.path.join(root, output)
os.makedirs(inputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)

for video in glob.glob(inputDir + '**/*', recursive=True):
    if os.path.isfile(video):
        probe = ffmpeg.probe(video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        num_frames = int(video_info['nb_frames'])
        print('IN: ' + video)

        ofilename = os.path.basename(os.path.splitext(video)[0]) + '_%03d.png'

        top = os.path.basename(os.path.dirname(input))
        new_top_index = video.find(top) + len(top) + 1
        new_last_index = video.find(os.path.basename(video))

        ofoldername = video[new_top_index:new_last_index]
        ofilepath = os.path.join(outputDir, ofoldername)
        os.makedirs(ofilepath, exist_ok=True)
        ofullpath = os.path.join(ofilepath, ofilename)
        print('OUT: ' + ofullpath)
        ffmpeg.input(video).output(ofullpath, r=1.5, vframes=num_frames, format='image2').run(capture_stdout=True)
