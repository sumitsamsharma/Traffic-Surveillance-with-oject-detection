
import datetime
import cv2
import numpy as np
from collections import deque
from deep_sort.deep_sort.detection import Detection


# Define some parameters
absence_counters = {}
acc_threshold = 0.4
max_cosine_distance = 0.4
nn_budget = None
points = [deque(maxlen=32) for _ in range(1000)]

first_line = (200, 280)
first_line_end = (680, 280)

class_counters = {
    'car': 0,
    'bus': 0,
    'motorcycle': 0,
    'truck': 0,
    'threewheel': 0,
    'van': 0,
    'Hiace': 0,
    'Rickshaw': 0,
    'Tractor': 0,
    'vehicle': 0
}


def process_video(video_cap, writer, model, encoder, tracker, class_names, colors):
    location = [deque(maxlen=32) for _ in range(1000)]
    tracker_list = []

    while True:
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        cv2.line(frame, first_line, first_line_end, (0, 255, 0), 12)

        if not ret:
            print("End of the video file...")
            break

        results = model(frame)

        for result in results:
            bboxes = []
            accuracy_scores = []
            class_ids = []

            for data in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = data
                x, y, w, h, class_id = map(int, (x1, y1, x2 - x1, y2 - y1, class_id))

                if confidence > acc_threshold:
                    bboxes.append([x, y, w, h])
                    accuracy_scores.append(confidence)
                    class_ids.append(class_id)

        names = [class_names[class_id] for class_id in class_ids]
        features = encoder(frame, bboxes)
        dets = []

        for bbox, conf, class_name, feature in zip(bboxes, accuracy_scores, names, features):
            dets.append(Detection(bbox, conf, class_name, feature))

        tracker.predict()
        tracker.update(dets)
        active_track_ids = set()

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            active_track_ids.add(track.track_id)
            bbox = track.to_tlbr()
            track_id = track.track_id
            class_name = track.get_class()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            class_id = class_names.index(class_name)
            color = colors[class_id]
            blue, green, red = int(color[0]), int(color[1]), int(color[2])
            text = str(track_id) + " - " + class_name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (blue, green, red), 3)
            cv2.rectangle(frame, (x1 - 1, y1 - 20),
                          (x1 + len(text) * 12, y1), (blue, green, red), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            location[track_id].append((center_x, center_y))

            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)

            for i in range(1, len(location[track_id])):
                current_loc = location[track_id][i - 1]
                prev_loc = location[track_id][i]
                if current_loc is None or prev_loc is None:
                    continue
                cv2.line(frame, (current_loc), (prev_loc), (0, 255, 0), 2)

            last_point_x = location[track_id][0][0]
            last_point_y = location[track_id][0][1]
            cv2.circle(frame, (int(last_point_x), int(last_point_y)), 4, (255, 0, 255), -1)

            if (class_name, track_id) not in tracker_list:
                if center_y > first_line[1] > last_point_y and first_line[0] < center_x < first_line_end[0]:
                    class_name = track.get_class()
                    tracker_list.append((class_name, track_id))
                    if class_name in class_counters:
                        class_counters[class_name] += 1

            for class_name, id in tracker_list:
                if id not in active_track_ids:
                    if id not in absence_counters:
                        absence_counters[id] = 1
                    else:
                        absence_counters[id] += 1
                else:
                    absence_counters[id] = 0

                if absence_counters.get(id, 0) >= 40:
                    class_nam = [item[0] for item in tracker_list if item[1] == id]
                    class_name = class_nam[0]
                    if class_name in class_counters:
                        class_counters[class_name] -= 1
                    else:
                        class_counters[class_name] -= 1

                    absence_counters[id] = 0
                    tracker_list = [item for item in tracker_list if item[1] != id]

        cv2.fillPoly(frame, [np.array([(0, 0), (0, 120), (200, 120), (200, 0)])], (0, 0, 0))
        cv2.putText(frame, f"Car Count: {class_counters['car']}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Bus Count: {class_counters['bus']}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, f"Motorcycle Count: {class_counters['motorcycle']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 255), 2)
        cv2.putText(frame, f"Truck Count: {class_counters['truck']}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2)
        cv2.putText(frame, f"Total vehicles: {len(active_track_ids)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0),2)

        cv2.imshow("Output", frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord("q"):
            break
