import cv2


# Initialize the video capture and the video writer objects
def initialize_video():
    video_cap = cv2.VideoCapture("traffic.mp4")
    writer = create_video_writer(video_cap, "output.mp4")
    return video_cap, writer


# Helper function to create a video writer
def create_video_writer(video_cap, output_file):
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_file, fourcc, 20, (frame_width, frame_height))
    return writer

