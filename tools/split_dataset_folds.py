import sys
import os
import glob
from shutil import copyfile


inputDir = str(sys.argv[1])  # input
dataDir = str(sys.argv[2])  # output
outputDir = str(sys.argv[3])  # output
test = os.path.join(outputDir, "test")
train = os.path.join(outputDir, "train")
os.makedirs(test, exist_ok=True)
os.makedirs(train, exist_ok=True)

for f in glob.glob(os.path.join(inputDir, "**/*.txt"), recursive=True):
    # print(f)
    ftxt = open(f, "r")
    fold = train

    if "test" in f:
        fold = test

    for line in ftxt:
        if "/" in f:
            path = line.replace("/", "_")
        else:
            path = line.replace("\\", "_")

        img = "release_1_" + path.rstrip("\n")
        print("Reading: " + img)

        if "attack" in img:
            inPath = os.path.join(dataDir, "attack")
            outPath = os.path.join(fold, "attack")
        else:
            inPath = os.path.join(dataDir, "real")
            outPath = os.path.join(fold, "real")

        os.makedirs(outPath, exist_ok=True)
        sub = "00"

        for i in range(48):
            if i >= 10:
                sub = "0"
            if i != 0:
                file = img + "_" + sub + str(i) + ".png"
                inFile = os.path.join(inPath, file)
                outFile = os.path.join(outPath, file)
                print("In: " + inFile)
                print("Out: " + outFile + "\n")
                copyfile(inFile, outFile)






