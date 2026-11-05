# Point Cloud Classifier: Car vs Pedestrian

A simple machine learning pipeline that looks at a 3D point cloud (the kind of data radar or LiDAR produces) and predicts whether the object is a car or a pedestrian. It includes a local web app where you upload points, see them in 3D, and get a prediction.

Pipeline:  points (CSV) → 4 features → Decision Tree → prediction

## How it works
1. Data: 1,000 simulated objects (500 cars, 500 pedestrians) with realistic sizes, sensor noise, and about 30% of cars partly hidden.
2. Features:  each point cloud is measured as length, width, height and number of points.
3. Model: a Decision Tree (scikit-learn), trained on 80% of the data and tested on the other 20%.
4. App: a Streamlit page that shows the points in 3D and the model's prediction.

## Results
- 100% accuracy on 200 unseen test examples.
- This is high because the simulated data is clean; real radar data would be harder.
- My first model only used length and called a partly hidden car a pedestrian. Adding hidden cars to the training data fixed this: the model learned to use width instead.
## Run it
