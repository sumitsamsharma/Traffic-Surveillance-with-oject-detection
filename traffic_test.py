import cv2


def test_traffic(roi_vehicles, frame, traffic_threshold):
    if len(roi_vehicles) > traffic_threshold:
            traffic_text = "Traffic"
            cv2.putText(frame, traffic_text, (190, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)
