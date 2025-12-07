import cv2
import os
from preprocessing import abocnn_preprocessing

root = r"D:\Projects\oneclass_abocnn_project\data\rice4class"

print("Testing images...")

bad_images = []

for cls in os.listdir(root):
    cls_path = os.path.join(root, cls)
    for file in os.listdir(cls_path):
        path = os.path.join(cls_path, file)

        img = cv2.imread(path)

        # unreadable image
        if img is None:
            bad_images.append(path)
            continue

        try:
            _ = abocnn_preprocessing(img)
        except Exception as e:
            bad_images.append((path, str(e)))

print("\nBad images found:", len(bad_images))
for item in bad_images:
    print(item)
