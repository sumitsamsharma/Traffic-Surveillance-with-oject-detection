from running_code.video_initialize import initialize_video
from running_code.fetch_model_and_tracker import initialize_deep_sort_tracker, initialize_yolo_model
from running_code.process_frames_and_detect import process_video
from running_code.colour_for_bounding_boxes import create_random_colors
import cv2
import os
os.getcwd()


def main():
    video_cap, writer = initialize_video()
    model = initialize_yolo_model()
    encoder, tracker = initialize_deep_sort_tracker()
    class_names = ['bus', 'car', 'motorcycle', 'truck', 'threewheel', 'van', 'Hiace', 'Rickshaw', 'Tractor', 'vehicle']
    colors = create_random_colors(class_names)
    process_video(video_cap, writer, model, encoder, tracker, class_names, colors)
    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
