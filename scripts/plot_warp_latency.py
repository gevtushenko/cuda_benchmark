#!/bin/env python3

import matplotlib.pyplot as plt
import csv
import sys

if len(sys.argv) < 2:
    print("The script is called with at least one argument")
    sys.exit(-1)

data = []
with open(sys.argv[1]) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for row in csv_reader:
        data.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

last_warp = 0
for warp in data:
    if warp[1] == 0:
        x_values = [warp[2], warp[3]]
        y_values = [last_warp, last_warp]

        last_warp += 1

        plt.plot(x_values, y_values, color="blue")

plt.xlabel("latency")
plt.ylabel("warp id")
plt.xticks([], [])
plt.show()
