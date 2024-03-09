import cv2
import json
from pprint import pprint

file_name = 'ab4c6d14-210514_183025_Garmin_Oregon7xx_DSC03253_jpg.rf.a5c164066e8ad8b6d965c44630f17607.jpg'
image = cv2.imread('Tree detection/train/{}'.format(file_name))

annotation_file_path = 'Tree detection/train/_annotations.coco.json'
with open(annotation_file_path, "r") as json_file:
    json_data = json.load(json_file)
    annotations = json_data['annotations']

pprint(annotations)

# image_id = None
# for img in json_data['images']:
#     print(img['file_name'])
#     if img['file_name'] == file_name:
#         image_id = img['id']
#         break
#
# if image_id is not None:
#     for ann in annotations:
#         if ann['image_id'] == image_id:
#             bbox = ann['bbox']
#             xmin, ymin, width, height = map(int, bbox)
#             cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0, 255, 0), 2)
#
# cv2.imshow('Image with Bounding Boxes', image)
# cv2.waitKey(0)

# image
# {'date_captured': '2024-03-06T15:53:03+00:00',
#  'file_name': '6b19bfca-230930_170720_Garmin_Oregon7xx_DSC04277_jpg.rf.e8fdee3014247122d6a73517fb56800b.jpg',
#  'height': 640,
#  'id': 1051,
#  'license': 1,
#  'width': 640},
#
# annotate
# {'area': 32864,
#   'bbox': [5, 314, 208, 158],
#   'category_id': 1,
#   'id': 1354,
#   'image_id': 1052,
#   'iscrowd': 0,
#   'segmentation': []}