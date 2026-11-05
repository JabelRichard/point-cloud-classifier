"""
Generates test point clouds for the app.

Writes 15 CSVs (x, y, z) to test_files/:
  - 5 cars
  - 5 partially occluded cars
  - 5 pedestrians

Same format as the samples, so they can be loaded straight into the app.
"""

import os
import numpy as np
import pandas as pd


def make_object(kind):
    # Pick a random size (in metres) and a random number of points
    if kind == "car":
        length = np.random.uniform(3.8, 5.0)
        width = np.random.uniform(1.6, 2.0)
        height = np.random.uniform(1.3, 1.8)
        n = np.random.randint(150, 300)
    elif kind == "hidden_car":
        # Same as a normal car, but something's in the way so we only see a short bit of it
        length = np.random.uniform(1.0, 2.5)
        width = np.random.uniform(1.6, 2.0)
        height = np.random.uniform(1.3, 1.8)
        n = np.random.randint(150, 300)
    else:  # pedestrian
        length = np.random.uniform(0.3, 0.7)
        width = np.random.uniform(0.3, 0.7)
        height = np.random.uniform(1.5, 1.9)
        n = np.random.randint(50, 120)

    # Throw the points into a box that size, then shake them a little
    # because real sensors are never spot on
    x = np.random.uniform(0, length, n)
    y = np.random.uniform(0, width, n)
    z = np.random.uniform(0, height, n)
    noise = np.random.normal(0, 0.05, (n, 3))
    return np.column_stack([x, y, z]) + noise


# Create the folder first (no fuss if it's already there)
os.makedirs("test_files", exist_ok=True)

# Five of each kind, saved as test_files/car_1.csv, car_2.csv and so on
for kind in ["car", "hidden_car", "pedestrian"]:
    for i in range(1, 6):
        points = pd.DataFrame(make_object(kind), columns=["x", "y", "z"])
        filename = f"test_files/{kind}_{i}.csv"
        points.to_csv(filename, index=False)
        print("Saved", filename)
