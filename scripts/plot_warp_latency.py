#!/bin/env python3

import matplotlib.pyplot as plt
import report_loader
import sys

if len(sys.argv) < 2:
    print("The script is called with at least one argument")
    sys.exit(-1)


for filename in sys.argv[1:len(sys.argv)]:
    report = report_loader.load_report(filename)

    last_warp = 0
    for warp in report.get_records_for_sm(0):
        x_values = [warp.start_time, warp.end_time]
        y_values = [last_warp, last_warp]

        last_warp += 1

        plt.plot(x_values, y_values, color="blue")

plt.xlabel("latency")
plt.ylabel("warp id")
plt.xticks([], [])
plt.show()
