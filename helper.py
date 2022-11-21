import os
import numpy as np
import cv2
import datetime
import xml.etree.ElementTree as voc

from pathlib import Path


def detect(img, model):

    result = model(img)
    detections = result.pandas().xyxy
    detectionsArray = np.array(detections)[0]
    return detectionsArray

# for COCO


def get_metadata(image, img_count, filename):
    date = datetime.datetime.now()
    return {"id": img_count,
            "license": 1,
            "file_name": filename,
            "height": image.shape[1],
            "width": image.shape[0],
            "date_captured": f"{date.year}/{date.month}/{date.day}"
            }


def get_annotations(pred, count, image_id):
    xmin, ymin = round(pred[0]), round(pred[1])
    width, height = round(pred[2])-xmin, round(pred[3])-ymin
    area = width*height
    return {"id": count, "image_id": image_id, "category_id": pred[-2], "bbox": [xmin, ymin, width, height], "area": area, "segmentation": [], "iscrowd": 0}

# for PASCAL VOC


def pred_to_xml(image, array, filepath):
    root = voc.Element("annotation")

    folder = voc.Element("folder")
    folder.text = ""
    root.append(folder)

    filename = voc.Element("filename")
    filename.text = str(os.path.basename(filepath))
    root.append(filename)

    path = voc.Element("path")
    path.text = str(os.path.basename(filepath))
    root.append(path)

    source = voc.Element("source")
    database = voc.SubElement(source, "database")
    database.text = "Unknown"
    root.append(source)

    size = voc.Element("size")
    width = voc.SubElement(size, "width")
    width = image.shape[0]
    height = voc.SubElement(size, "height")
    height = image.shape[1]
    depth = voc.SubElement(size, "depth")
    depth = image.shape[2]
    root.append(size)

    segmented = voc.Element("segmented")
    segmented.text = '0'
    root.append(segmented)

    for i in range(len(array)):
        object = voc.Element("object")
        name = voc.SubElement(object, "name")
        name.text = array[i][-1]
        pose = voc.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = voc.SubElement(object, "truncated")
        truncated.text = "0"
        difficult = voc.SubElement(object, "difficult")
        difficult.text = "0"
        occluded = voc.SubElement(object, "occluded")
        occluded.text = "0"
        bndbox = voc.SubElement(object, "bndbox")
        xmin = voc.SubElement(bndbox, "xmin")
        xmin.text = str(round(array[i][0]))
        xmax = voc.SubElement(bndbox, "xmax")
        xmax.text = str(round(array[i][2]))
        ymin = voc.SubElement(bndbox, "ymin")
        ymin.text = str(round(array[i][1]))
        ymax = voc.SubElement(bndbox, "ymax")
        ymax.text = str(round(array[i][3]))
        root.append(object)
    tree = voc.ElementTree(root)

    with open(f"images/{Path(filepath).stem}.xml", "wb") as f:
        tree.write(f)
