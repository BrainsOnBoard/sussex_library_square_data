import seaborn as sns
import sys

presentation = "presentation" in sys.argv[1:]

def remove_axis_junk(axis):
    # Turn off grid
    axis.xaxis.grid(False)
    axis.yaxis.grid(False)

# Set the plotting style
if presentation:
    sns.set(context="talk")
    sns.set_style("whitegrid", {"font.family":"sans-serif", "font.sans-serif":"Verdana"})
else:
    sns.set(context="paper")
    sns.set_style("whitegrid", {"font.family":"serif", "font.serif":"Times New Roman"})

# **HACK** fix bug with markers
sns.set_context(rc={"lines.markeredgewidth": 1.0})

gutter_width = 0.38
column_width = 3.31
double_column_width = (column_width * 2.0) + gutter_width
