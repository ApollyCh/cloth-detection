import os
import json
import cv2

INFO = {
    "description": "DeepFashion2",
    "url": "https://github.com/switchablenorms/DeepFashion2",
}

CATEGORIES = [
    {"id": 1, "name": "short sleeved shirt"},
    {"id": 2, "name": "long sleeved shirt"},
    {"id": 3, "name": "short sleeved outwear"},
    {"id": 4, "name": "long sleeved outwear"},
    {"id": 5, "name": "vest"},
    {"id": 6, "name": "sling"},
    {"id": 7, "name": "shorts"},
    {"id": 8, "name": "trousers"},
    {"id": 9, "name": "skirt"},
    {"id": 10, "name": "short sleeved dress"},
    {"id": 11, "name": "long sleeved dress"},
    {"id": 12, "name": "vest dress"},
    {"id": 13, "name": "sling dress"},
]


def get_image_info(file_name, image_id, width=0, height=0):
    return {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id,
    }


def get_annotation_info(ann_id, image_id, category_id, bbox, segmentation):
    return {
        "mask": segmentation,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": ann_id,
    }


def create_coco_json(images, annotations, output_path):
    data = {
        "info": INFO,
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    with open(output_path, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    img_dir = '../DeepFashion2/some_train/image'
    ann_dir = '../DeepFashion2/some_train/annos'
    output_path = '../DeepFashion2/some_train/coco_train.json'

    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for img_file in os.listdir(img_dir):

        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        images.append(get_image_info(img_file, image_id, width, height))

        ann_file = img_file.replace('.jpg', '.json')
        ann_path = os.path.join(ann_dir, ann_file)

        with open(ann_path, 'r') as f:
            ann_data = json.load(f)

        for ann in ann_data.keys():
            if ann in ['source', 'pair_id']:
                continue

            category_id = int(ann_data[ann]['category_id'])
            bbox = ann_data[ann]['bounding_box']
            segmentation = ann_data[ann]['segmentation']

            annotations.append(get_annotation_info(ann_id, image_id, category_id, bbox, segmentation))
            ann_id += 1

        image_id += 1

    create_coco_json(images, annotations, output_path)
