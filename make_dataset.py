"""
Builds the training data.
I fake 1,000 objects, 500 cars and 500 pedestrians, as clouds of 3D points,
roughly what a radar would hand you. To keep things honest, about 30% of the
cars are partly hidden (behind another car, a wall, whatever), so you only
see a chunk of their length.
For every object I measure four things: length, width, height and how many
points it has. Those go into dataset.csv. I also save one example car and
one example pedestrian (sample_car.csv, sample_pedestrian.csv) so there's
something to try in the app straight away.
"""

import pandas as pd
import numpy as np


def make_object(kind):

    # Give it a random size (metres) and a random number of points
    if kind == "car":
        length = np.random.uniform(3.8, 5.0)
        width = np.random.uniform(1.6, 2.0)
        height = np.random.uniform(1.3, 1.8)
        n = np.random.randint(150, 300)
    else:  # pedestrian
        length = np.random.uniform(0.3, 0.7)
        width = np.random.uniform(0.3, 0.7)
        height = np.random.uniform(1.5, 1.9)
        n = np.random.randint(50, 120)

    # Roughly 1 in 3 cars is partly blocked, so we only catch a short bit of it
    if kind == "car" and np.random.rand() < 0.3:
        length = np.random.uniform(1.0, 2.5)

    # Scatter the points around inside a box that size
    x = np.random.uniform(0, length, n)
    y = np.random.uniform(0, width, n)
    z = np.random.uniform(0, height, n)

    # Real sensors are never perfect, so jiggle every point a little
    noise = np.random.normal(0, 0.05, (n, 3))
    points = np.column_stack([x, y, z]) + noise
    return points


def get_features(points):
    """Work out length, width, height and point count for one point cloud."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # How big it is in each direction: furthest point minus nearest point
    return {
        "length": x.max() - x.min(),
        "width": y.max() - y.min(),
        "height": z.max() - z.min(),
        "num_points": len(points),
    }


# One row per object: its measurements plus what it actually is
rows = []

for i in range(500):
    features = get_features(make_object("car"))
    features["label"] = "car"
    rows.append(features)

    features = get_features(make_object("pedestrian"))
    features["label"] = "pedestrian"
    rows.append(features)

# Write it all out
df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)

print(df.head())
print("Total rows:", len(df))

# And one of each to play with in the app
car_points = pd.DataFrame(make_object("car"), columns=["x", "y", "z"])
car_points.to_csv("sample_car.csv", index=False)

ped_points = pd.DataFrame(make_object("pedestrian"), columns=["x", "y", "z"])
ped_points.to_csv("sample_pedestrian.csv", index=False)

print("Sample files saved!")
