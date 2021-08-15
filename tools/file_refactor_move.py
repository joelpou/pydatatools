import sys
import os
import glob
from shutil import copyfile

inputDir = str(sys.argv[1])
outputDir = str(sys.argv[2])  # output path to dump all images converted
f_ext = str(sys.argv[3])

print("Start test...")

for f in glob.glob(os.path.join(inputDir, "**/*." + f_ext), recursive=True):
    print("Refactoring file: {}".format(f))
    top = os.path.basename(os.path.dirname(inputDir))
    new_top_index = f.find(top) + len(top) + 1
    new_last_index = f.find(os.path.basename(f))

    path = f[new_top_index:new_last_index]

    if "/" in path:
        path = path.replace("/", "_")
    else:
        path = path.replace("\\", "_")

    name = path + os.path.basename(f)

    if "attack" in name:
        dst = os.path.join(outputDir, "attack")
    else:
        dst = os.path.join(outputDir, "real")

    dstFile = os.path.join(dst, name)
    os.makedirs(dst, exist_ok=True)
    copyfile(f, dstFile)
