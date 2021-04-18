#!/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import report_loader
import sys

if len(sys.argv) < 2:
    print("The script is called with at least one argument")
    sys.exit(-1)


for filename in sys.argv[1:len(sys.argv)]:
    report = report_loader.load_report(filename)

    durations = []
    for warp in report.get_records():
        durations.append(warp.duration())

    sns.distplot(durations, rug=True)

plt.show()
