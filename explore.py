"""
My first experiment: looking at a fake point cloud in 3D.
I scatter 200 random points inside a car-sized box and plot them so I can
spin them around and see what a point cloud actually looks like. The plot
gets saved to car.html and opens in your browser.
"""

import numpy as np
import plotly.express as px

# 200 random points inside a box the size of a car (in metres)
x = np.random.uniform(0, 4.5, 200)   # length: 4.5 m
y = np.random.uniform(0, 1.8, 200)   # width: 1.8 m
z = np.random.uniform(0, 1.5, 200)   # height: 1.5 m

# Draw the points in 3D and save the plot as an HTML page
fig = px.scatter_3d(x=x, y=y, z=z, title="A simulated car")
fig.update_traces(marker=dict(size=3))
fig.update_layout(scene_aspectmode="data")
fig.write_html("car.html", auto_open=True)
