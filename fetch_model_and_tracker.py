from ultralytics import YOLO
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching

from deep_sort.tools import generate_detections as gdet
max_cosine_distance = 0.4
nn_budget = None


def initialize_yolo_model():
    model = YOLO('dataset/roboflow/runs/detect/my_model34/weights/last.pt')
    return model


# Initialize the deep sort tracker
def initialize_deep_sort_tracker():
    model_filename = "config/mars-small128.pb"
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker
