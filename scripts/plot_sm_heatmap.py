#!/bin/env python3

import matplotlib.pyplot as plt
import report_loader
import math
import sys

if len(sys.argv) < 2:
    print("The script is called with at least one argument")
    sys.exit(-1)


def reshape(data):
    result = []
    side_size = int(math.sqrt(len(data)))

    last_id = 0
    for i in range(side_size):
        row = []
        for j in range(side_size):
            row.append(data[last_id])
            last_id += 1
        result.append(row)

    return result


for filename in sys.argv[1:len(sys.argv)]:
    report = report_loader.load_report(filename)

    total_sm_time = dict()

    for warp in report.get_records():
        total_sm_time[warp.sm_id] = warp.duration()

    to_plot = reshape(list(total_sm_time.values()))
    plt.imshow(to_plot, cmap="hot") #, vmin=0)

plt.show()
