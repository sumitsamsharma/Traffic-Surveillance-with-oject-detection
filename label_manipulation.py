import os

label_dir = 'C:/Users/SharmaS/Downloads/vehicles.v2-release.yolov8/train/labels'
category_mapping = {
2:0,
3:0,
6:0,
1:3, 5:3, 7:3, 8:3, 9:3, 10:3, 11:3, 4:1,
}

# Iterate through label files
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Modify the labels
        modified_lines = []
        for line in lines:
            label_idx, *rest = line.split()
            modified_label_idx = category_mapping.get(int(label_idx), label_idx)
            modified_line = f'{modified_label_idx} ' + ' '.join(rest) + '\n'
            modified_lines.append(modified_line)

        # Save the modified labels with the same filename
        with open(filepath, 'w') as f:
            f.writelines(modified_lines)
