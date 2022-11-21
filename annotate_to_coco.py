import os
import datetime
import platform
import argparse
import json
from glob import glob

import cv2
import numpy as np
import torch

from helper import get_annotations, get_metadata, detect

model = torch.hub.load("ultralytics/yolov5", "custom", path="model/best.pt")
model.conf = 0.6
model.iou = 0.2


def auto_annotate_coco(filepath, format):

    current_date = datetime.datetime.now()
    year = current_date.year
    month = current_date.month
    day = current_date.day
    filepath = filepath
    format = format
    output = {"info": {
        "year": year,
        "version": "1.0",
        "description": "Annotation using pretrained models",
        "contributor": platform.node(),
        "url": "placeholder.com",
        "date_created": f"{year}/{month}/{day}",

        "licenses": [
                        {
                            "url": "https://creativecommons.org/licenses/by/4.0/",
                            "id": 1,
                            "name": "CC BY 4.0",
                        },
        ],
        "categories": [{"id": 0, "name": "pipes", "supercategory": "none"}, {"id": 1, "name": "Pipe", "supercategory": "pipes"}],
        "images": [],
        "annotations": [],
    }
    }

    img_count = -1
    annotate_count = -1

    for i in glob(filepath + f"/*.{format}"):
        img_count += 1
        image = cv2.imread(i)
        output["info"]["images"].append(get_metadata(
            image, img_count, os.path.basename(i)))
        pred = detect(image, model)
        print(i)

        for j in range(len(pred)):
            annotate_count += 1
            output["info"]["annotations"].append(get_annotations(
                pred[j], annotate_count, os.path.basename(i)))

    out_file = open(f'{filepath}/labels.json', 'w+')

    return json.dump(output, out_file)


def parse_opt():
    parser = argparse.ArgumentParser(
        description='params for Filepath and image format')
    parser.add_argument('--filepath', type=str, default='image/',
                        help="filepath containing images to be annotated relative to the main.py")
    parser.add_argument('--format', type=str, default='jpg',
                        help='image format, please make them uniform before running this script')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_opt()
    print(args)
    auto_annotate_coco(**vars(args))
