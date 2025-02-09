import os
import argparse
import cv2
import numpy as np

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, default='test1.jpg')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_models/best.pt')
    parser.add_argument('-t', '--conf_thres', type=float, default=0.2)
    args = parser.parse_args()
    return args

def test(args):
    classes = ['Tree']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn().to(device)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('No checkpoint')
        exit(0)

    # normalization
    ori_img = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # transpose
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device).float()
    prediction = model(image)
    boxes = prediction[0]["boxes"]
    labels = prediction[0]["labels"]
    scores = prediction[0]["scores"]

    for box, label, score in zip(boxes, labels, scores):
        if score >= args.conf_thres:
            xmin, ymin, width, height = map(int, box)
            xmax = xmin + width
            ymax = ymin + height
            cv2.rectangle(ori_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(ori_img, classes[label-1] + " {:.2f}".format(score), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 2)
        cv2.imshow("test.jpg", ori_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    args = parse_args()
    test(args)

