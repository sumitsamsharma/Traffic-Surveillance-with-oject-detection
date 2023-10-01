import os


def delete_unmatched_label_img():

    labels_folder = 'dataset/roboflow/train/labels'
    images_folder = 'dataset/roboflow/train/images'

    label_files = [file for file in os.listdir(labels_folder) if file.endswith('.txt')]
    image_files = [file for file in os.listdir(images_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

    print(len(label_files), '  ', len(image_files))

    labels_without_images = [label for label in label_files if
                             label[:-4] + '.jpg' not in image_files and label[:-4] + '.png' not in image_files and label[:-4] + '.jpeg' not in image_files]

    images_without_labels = [image for image in image_files if image[:-4] + '.txt' not in label_files]

    for label in labels_without_images:
        label_path = os.path.join(labels_folder, label)
        os.remove(label_path)
        print(f"Deleted label: {label}")

    for image in images_without_labels:
        image_path = os.path.join(images_folder, image)
        os.remove(image_path)
        print(f"Deleted image: {image}")

    if not images_without_labels and not labels_without_images:
        print(" All labels have corresponding images, and all images have corresponding labels.")
    else:
        print(" There are unmatched labels or images.")
        print("Unmatched images:", images_without_labels)
        print("Unmatched labels:", labels_without_images)


