from ultralytics import YOLO
import os

# Loading a pretrained YOLO model for training on custom data
model = YOLO('yolov8n.pt')
#model = YOLO('C:/MSc Project 2/dataset/roboflow/runs/detect/my_model32/weights/last.pt')

# training the model for 10 epochs
model.train(data='data_robo.yaml', epochs=10, name='my_model')

# checking the model performance on validation data
model.val(data='data_robo.yaml')

# saving the model
path = model.export(format="onnx")
