"""
Hidden car test. Picture a car parked behind a wall where you only see
the last 1.5 m or so. Going by length it looks like a pedestrian, but
the width is still car width. Want to see if the classifier falls for it.

Dumps the points into tricky_hidden_car.csv, just load that in the app.
"""

import numpy as np
import pandas as pd

# 200 random points in a box, short like a person but wide like a car
n = 200
x = np.random.uniform(0, 1.5, n)   # only the visible part
y = np.random.uniform(0, 1.8, n)   # car width
z = np.random.uniform(0, 1.5, n)   # car height

pd.DataFrame({"x": x, "y": y, "z": z}).to_csv("tricky_hidden_car.csv", index=False)
print("Tricky file saved!")
