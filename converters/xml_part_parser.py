"""
Script that parses out specific annotation parts from xml
"""

import sys

input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

parts_wanted = list(range(36, 48))


def get_part_index(l):
    i = l.find("'")
    pnum = line[(i + 1):(i + 3)]
    return int(pnum)


with open(input_file, 'r') as f:
    lines = f.readlines()

with open(output_file, 'w') as f:
    for line in lines:
        if 'part' in line:
            p = get_part_index(line)
            if p in parts_wanted:
                f.write(line)
        else:
            f.write(line)
