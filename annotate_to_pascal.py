import argparse
import torch
import cv2
from glob import glob
from helper import detect, pred_to_xml


def main(filepath, format):
    model = torch.hub.load('ultralytics/yolov5',
                           'custom', path='model/best.pt')
    model.conf = 0.5
    model.iou = 0.2

    filepath = 'images'
    format = 'jpg'

    for i in glob(filepath + f"/*.{format}"):
        image = cv2.imread(i)
        array = detect(image, model)
        pred_to_xml(image, array, i)


def parse_opt():
    parser = argparse.ArgumentParser(
        description='params for Filepath and image format')
    parser.add_argument('--filepath', type=str, default='images',
                        help="filepath containing images to be annotated relative to the main.py")
    parser.add_argument('--format', type=str, default='jpg',
                        help='image format, please make them uniform before running this script')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_opt()
    main(**vars(args))
