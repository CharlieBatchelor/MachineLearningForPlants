from matplotlib import pyplot as plt
import numpy as np
import argparse

# rate = [0.32,1.45,4.36,10,22.3,45]
# thresh = [200, 150, 120, 100, 80, 60]

# rate = [25,43,80]
# thresh = [60, 50, 40]

# rate = [114,119,124,133,150,185,181,3400,177,4300,179]
# thresh = [120,110,100,90,75,60,45,30,35,30,40]

first = [0, 0, 0.01, 0.01, 0.015, 0.017, 0.022, 0.028, 0.037, 0.05, 0.053, 0.057, 0.06, 0.071, 0.087, 0.094, 0.1, 0.13, 0.15, 0.18]
second = [0, 0, 0.01, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.05, 0.05, 0.06, 0.07, 0.085, 0.095, 0.11, 0.12, 0.14, 0.17]
third = [0, 0.01, 0.01, 0.02, 0.021, 0.022, 0.023, 0.03, 0.04, 0.05, 0.055, 0.059, 0.065, 0.07, 0.09, 0.095, 0.1, 0.12, 0.15, 0.16]

# plt.plot(thresh, rate)
plt.plot(first, label="Recipe 1")
plt.plot(second, label="Recipe 2")
plt.plot(third, label="Recipe 3")
# plt.yscale('log')
legend_properties = {'weight':'bold'}
plt.legend(prop=legend_properties)
plt.grid('both')
plt.ylabel("Relative Area", fontweight='bold')
plt.xlabel("Relative Time", fontweight='bold')
plt.title('Relative Area Growth Metric', fontweight='bold')
# plt.ylabel("Total TPG Rate kHz", fontweight='bold')
# plt.xlabel("TPG Static Threshold", fontweight='bold')
# plt.title('Typical RMS - 15', fontweight='bold')
plt.show()