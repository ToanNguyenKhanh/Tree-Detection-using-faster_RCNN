from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
import torch
import cv2
import argparse
import os
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Test faster-rcnn")
    parser.add_argument("--video_path", "-i", type=str, default="test_video4.mp4")
    parser.add_argument("--out_path", "-o", type=str, default="output/video/video_output.mp4")
    parser.add_argument('-c', '--checkpoint_path', type=str, default='trained_models/best.pt')
    parser.add_argument("--conf_thres", "-t", type=float, default=0.2)
    args = parser.parse_args()
    return args


def test(args):
    classes = ['Tree']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_320_fpn().to(device)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
    else:
        print('No checkpoint')
        exit(0)
    cap = cv2.VideoCapture(args.video_path)
    out = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()[None, :, :, :].to(device)
        prediction = model(image)
        boxes = prediction[0]["boxes"]
        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]
        for box, label, score in zip(boxes, labels, scores):
            if score >= args.conf_thres:
                xmin, ymin, width, height = map(int, box)
                xmax = xmin + width
                ymax = ymin + height
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(frame, classes[label-1] + " {:.2f}".format(score), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 2)
        out.write(frame)
    cap.release()
    out.release()

if __name__ == '__main__':
    args = get_args()
    test(args)
