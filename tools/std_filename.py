import os
import sys
import glob


files_path = sys.argv[1]
# string = "BEGINNER'S GUIDE TO MEDITATION » for a positive & productive day (part 1) - YouTube - Google Chrome 2021-07-16 01-08-57_001.jpg"

for f in glob.glob(os.path.join(files_path, "*.csv")):
    basename = os.path.basename(f)
    stringx = basename.rsplit("_", 1)
    alphanumeric = ''.join(e for e in stringx[0] if e.isalnum()).lower()
    renamed = alphanumeric + "_" + stringx[1]
    renamed_path = os.path.join(os.path.dirname(f), renamed)
    print("Writing file: {}".format(renamed_path))
    os.rename(f, renamed_path)
