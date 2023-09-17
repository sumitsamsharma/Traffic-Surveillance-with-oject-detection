import numpy as np


# Create a list of random colors for drawing bounding boxes
def create_random_colors(class_names):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    return colors