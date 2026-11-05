"""Panoptic inference on one image: writes an overlay and prints the segments."""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image, ImageDraw

from data.data_aug import make_coco_transforms
from tools.common import load_pretrained

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--out", default="outputs/sample_panoptic.png")
ap.add_argument("--checkpoint", default="checkpoints/detr-r50-panoptic-00ce5173.pth")
ap.add_argument("--categories", default=os.path.expanduser("~/datasets/coco/annotations/panoptic_val2017.json"))
a = ap.parse_args()

names = {c["id"]: c["name"] for c in json.load(open(a.categories))["categories"]}
model, _, post, _ = load_pretrained(a.checkpoint)
model.eval()
im = Image.open(a.image).convert("RGB")
img, _ = make_coco_transforms("val")(im, None)
img = img.cuda()
with torch.no_grad():
    outputs = model([img])
    size = torch.as_tensor(img.shape[-2:]).unsqueeze(0).cuda()
    orig = torch.as_tensor([im.height, im.width]).unsqueeze(0).cuda()
    res = post["panoptic"](outputs, size, orig)[0]

ids = rgb2id(np.array(Image.open(io.BytesIO(res["png_string"]))))
rng = np.random.default_rng(0)
base = np.array(im.resize((ids.shape[1], ids.shape[0]))).astype(float)
canvas = base.copy()
labels = []
for s in res["segments_info"]:
    mask = ids == s["id"]
    if not mask.any():
        continue
    canvas[mask] = 0.45 * base[mask] + 0.55 * rng.integers(40, 255, 3)
    ys, xs = np.nonzero(mask)
    labels.append((int(xs.mean()), int(ys.mean()), names[s["category_id"]], int(mask.sum())))
    print(names[s["category_id"]], "thing" if s["isthing"] else "stuff", int(mask.sum()))
out = Image.fromarray(canvas.astype(np.uint8))
draw = ImageDraw.Draw(out)
for x, y, name, area in labels:
    if area > 800:
        draw.text((x, y), name, fill="white", stroke_width=2, stroke_fill="black", font_size=20)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
out.save(a.out)
