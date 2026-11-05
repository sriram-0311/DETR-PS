"""Score the repo's model code with the official checkpoint on COCO panoptic val2017."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from panopticapi.evaluation import pq_compute
from PIL import Image

from data.data_aug import make_coco_transforms
from tools.common import load_pretrained

ap = argparse.ArgumentParser()
ap.add_argument("--coco", default=os.path.expanduser("~/datasets/coco"))
ap.add_argument("--checkpoint", default="checkpoints/detr-r50-panoptic-00ce5173.pth")
ap.add_argument("--out", default="outputs/coco_val")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

coco = Path(a.coco)
gt_json = coco / "annotations/panoptic_val2017.json"
gt = json.load(open(gt_json))
images = sorted(gt["images"], key=lambda x: x["id"])
if a.limit:
    images = images[:a.limit]
out = Path(a.out)
pred_dir = out / "pred"
pred_dir.mkdir(parents=True, exist_ok=True)

model, _, post, _ = load_pretrained(a.checkpoint)
model.eval()
transform = make_coco_transforms("val")
preds, forward_ms, post_ms = [], [], []
t0 = time.time()
with torch.no_grad():
    for i, info in enumerate(images):
        img, _ = transform(Image.open(coco / "val2017" / info["file_name"]).convert("RGB"), None)
        img = img.cuda()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        outputs = model([img])
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        res = post["panoptic"](outputs, torch.as_tensor(img.shape[-2:]).unsqueeze(0).cuda(),
                               torch.as_tensor([info["height"], info["width"]]).unsqueeze(0).cuda())[0]
        t3 = time.perf_counter()
        forward_ms.append(1e3 * (t2 - t1))
        post_ms.append(1e3 * (t3 - t2))
        name = info["file_name"].replace(".jpg", ".png")
        (pred_dir / name).write_bytes(res["png_string"])
        preds.append({"image_id": info["id"], "file_name": name, "segments_info": res["segments_info"]})
        if i % 250 == 0:
            print(f"{i}/{len(images)} forward {forward_ms[-1]:.0f} ms, elapsed {time.time() - t0:.0f} s", flush=True)

pred_json = out / "pred.json"
json.dump({"annotations": preds}, open(pred_json, "w"))
if a.limit:
    subset = {**gt, "images": images, "annotations": [x for x in gt["annotations"] if x["image_id"] in {i["id"] for i in images}]}
    gt_json = out / "gt_subset.json"
    json.dump(subset, open(gt_json, "w"))
pq = pq_compute(str(gt_json), str(pred_json), str(coco / "annotations/panoptic_val2017"), str(pred_dir))
summary = {
    "n_images": len(images),
    "pq": {k: {m: 100 * pq[k][m] for m in ("pq", "sq", "rq")} | {"n": pq[k]["n"]} for k in ("All", "Things", "Stuff")},
    "forward_ms_median": statistics.median(forward_ms[10:]),
    "postprocess_ms_median": statistics.median(post_ms[10:]),
    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    "gpu": torch.cuda.get_device_name(0),
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
