"""
Script that parses out specific annotation parts from csv
"""

import sys
import os
import glob

input_dir = str(sys.argv[1])
output_dir = str(sys.argv[2])
os.makedirs(output_dir, exist_ok=True)

parts_wanted = list(range(36, 48))


def get_part_index(l):
    i = l.find(",")
    pnum = line[:i]
    return int(pnum)


for cs in glob.glob(os.path.join(input_dir, "*.csv")):
    print("Processing file: {}".format(cs))
    with open(cs, 'r') as f:
        lines = f.readlines()
        box_line = lines[-1]

    output_file = os.path.join(output_dir, os.path.basename(cs))
    with open(output_file, 'w') as f:
        for lnum, line in enumerate(lines):
            p = get_part_index(line)
            if p in parts_wanted:
                f.write(line)
        f.write(box_line)
