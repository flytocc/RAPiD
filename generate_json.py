import os
import json
from PIL import Image

ROOT = "Datasets/fisheye/HABBOF"
tasks = (
    "Meeting1",
    "Meeting2",
    "Lab1",
    "Lab2"
)

for task in tasks:
    images, annotations, categories = [], [], []
    image_id, ann_id = 0, 0
    cats = {}

    image_root = f'{ROOT}/{task}'
    for root, dirs, files in os.walk(image_root, followlinks=True):
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext != '.txt':
                continue

            img_name = filename + '.jpg'
            img_w, img_h = Image.open(os.path.join(image_root, img_name)).size

            image_id += 1
            images.append({
                "id": image_id,
                "file_name": img_name,
                "width": img_w, "height": img_h,
            })

            with open(os.path.join(image_root, file), "r") as f:
                for line in f.readlines():
                    # line: person 1029 1799 106 220 4
                    cat_name, *cxcywha = line.strip().split(" ")
                    cx, cy, w, h, a = list(map(float, cxcywha))

                    cat_id = cats.get(cat_name, None)
                    if cat_id is None:
                        cat_id = len(cats) + 1
                        categories.append({
                            "id": cat_id,
                            "name": cat_name,
                            "supercategory": cat_name,
                        })
                        cats[cat_name] = cat_id

                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "area": w * h,
                        "bbox": [cx, cy, w, h, a],
                        "category_id": cat_id,
                        "image_id": image_id,
                        "iscrowd": 0,
                        "segmentation": [],
                    })

    gt = {"images": images, "annotations": annotations, "categories": categories}
    json.dump(gt, open(os.path.join(ROOT, f"{task}.json"), "w"))
