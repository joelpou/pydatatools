import sys
import os
import glob

# change protocol txt files from input dir to new name with "_" and no ext

inputDir = str(sys.argv[1])  # input
outputDir = str(sys.argv[2])  # output

for f in glob.glob(os.path.join(inputDir, "**/*.txt"), recursive=True):
    print(f)

    if "/" in f:
        j = f.rfind("/")
    else:
        j = f.rfind("\\")

    newOut = f[len(inputDir):j + 1]
    newOut = os.path.join(outputDir, newOut)
    os.makedirs(newOut, exist_ok=True)

    ftxt = open(f, "r")
    fileOut = os.path.join(newOut, os.path.basename(f))
    fout = open(fileOut, "w")

    for line in ftxt:
        if "/" in line:
            path = line.replace("/", "_")
        else:
            path = line.replace("\\", "_")

        fileName = path[:path.rfind(".")]
        print(fileName)

        fout.write(fileName + "\n")

    ftxt.close()
    fout.close()
