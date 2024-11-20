import os
import json
from PIL import Image
from pycocotools.coco import COCO
from shapely.geometry import Polygon, MultiPolygon


def yolo_format(coco_annotation_path, image_folder, output_label_folder):
    coco = COCO(coco_annotation_path)

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_filename = img_info['file_name']
        img_width, img_height = img_info['width'], img_info['height']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        os.makedirs(output_label_folder, exist_ok=True)
        label_path = os.path.join(output_label_folder, label_filename)

        with open(label_path, 'w') as label_file:
            for ann in anns:
                if 'segmentation' in ann and ann['segmentation']:
                    if ann['iscrowd'] == 0:
                        class_id = ann['category_id'] - 1

                        segmentation = ann['segmentation'][0]
                        poly = Polygon([(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)])

                        normalized_coords = []
                        for x, y in poly.exterior.coords:
                            normalized_coords.append(x / img_width)
                            normalized_coords.append(y / img_height)

                        label_file.write(f"{class_id} " + " ".join(map(str, normalized_coords)) + "\n")
