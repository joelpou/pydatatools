import ffmpeg
import os
import sys
import glob

root = os.getcwd()  # root path
input = str(sys.argv[1])  # path where all videos are stored
output = str(sys.argv[2])  # path to dump video subfolders with frames
inputDir = os.path.join(root, input)
outputDir = os.path.join(root, output)
os.makedirs(inputDir, exist_ok=True)
os.makedirs(outputDir, exist_ok=True)

for video in glob.glob(inputDir + '**/*.avi', recursive=True):
    if os.path.isfile(video):
        print('IN: ' + video)
        # probe = ffmpeg.probe(video)
        # video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        # width = int(video_info['width'])
        # height = int(video_info['height'])
        fname = os.path.basename(os.path.splitext(video)[0])
        last = int(fname[-1])
        if last == 4 or last == 5:
            access_type = "spoof"
            stream_frame_rate = 0.1  # stream specifier -r (fps): https://ffmpeg.org/ffmpeg.html#Main-options. 0.1 gives 2 images
        elif last == 1:
            access_type = "real"
            stream_frame_rate = 1.3  # 1.3 gives 8 images
        else:
            print("skipping video...")
            continue

        # print("access type: " + access_type)
        ofilename = os.path.basename(os.path.splitext(video)[0]) + '_%03d.png'

        top = os.path.basename(os.path.dirname(input))
        new_top_index = video.find(top)
        new_last_index = video.find(os.path.basename(video))

        ofoldername = video[new_top_index:new_last_index]
        ofilepath = os.path.join(os.path.join(outputDir, ofoldername), access_type)
        os.makedirs(ofilepath, exist_ok=True)
        ofullpath = os.path.join(ofilepath, ofilename)
        print('OUT: ' + ofullpath)
        ffmpeg.input(video).output(ofullpath, r=stream_frame_rate, format='image2').run(
            capture_stdout=True)
