import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from pandas import read_csv

sns.set(context="paper")
sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times New Roman"})

data = read_csv("benchmark_results/output.csv", delimiter=",", skipinitialspace=True,
                converters={"RMSE":lambda s: float(s[:-4])})

# Group data by combinations of memory type and variant
bars = data.groupby(["memory type", "variant"])

# Build name and data arrays
names = ["%s\n%s" % (n[0], n[1]) for n, _ in bars]
data = [d["RMSE"].values for _, d in bars]



fig, axis = plt.subplots()
axis.boxplot(data)
axis.set_xticklabels(names, rotation=90, horizontalalignment="right", verticalalignment="center_baseline")

axis.set_ylabel("RMSE [degrees]")

axis.xaxis.grid(False)

fig.tight_layout()
plt.show()
