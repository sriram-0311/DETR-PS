"""Check that the repo's training path learns: overfit its class head on a few COCO val images."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from panopticapi.utils import rgb2id
from PIL import Image

from data.data_aug import make_coco_transforms
from evaluation.evaluator import collate_fn
from models.DETR import build_model
from tools.common import coco_panoptic_args
from training.train import train_one_epoch
from utils.utils import masks_to_boxes

N_IMAGES, EPOCHS = 16, 60
coco = Path(os.path.expanduser("~/datasets/coco"))
gt = json.load(open(coco / "annotations/panoptic_val2017.json"))
infos = {i["id"]: i for i in gt["images"]}
anns = [x for x in sorted(gt["annotations"], key=lambda x: x["image_id"]) if len(x["segments_info"]) <= 12][:N_IMAGES]
transform = make_coco_transforms("val")


class Tiny(torch.utils.data.Dataset):
    def __len__(self):
        return len(anns)

    def __getitem__(self, idx):
        ann = anns[idx]
        info = infos[ann["image_id"]]
        img = Image.open(coco / "val2017" / info["file_name"]).convert("RGB")
        ids = rgb2id(np.array(Image.open(coco / "annotations/panoptic_val2017" / ann["file_name"]), dtype=np.uint32))
        masks = torch.as_tensor(ids == np.array([s["id"] for s in ann["segments_info"]])[:, None, None], dtype=torch.uint8)
        target = {
            "masks": masks,
            "labels": torch.tensor([s["category_id"] for s in ann["segments_info"]]),
            "boxes": masks_to_boxes(masks),
            "image_id": torch.tensor([ann["image_id"]]),
            "size": torch.as_tensor([info["height"], info["width"]]),
            "orig_size": torch.as_tensor([info["height"], info["width"]]),
            "iscrowd": torch.tensor([s["iscrowd"] for s in ann["segments_info"]]),
            "area": torch.tensor([s["area"] for s in ann["segments_info"]]),
        }
        return transform(img, target)


args = coco_panoptic_args("cuda")
model, criterion, _ = build_model(args)
state = torch.load("checkpoints/detr-r50-panoptic-00ce5173.pth", map_location="cpu", weights_only=True)["model"]
# Reinitialise the class head, then train it on these images so the loss has something to learn.
state = {k: v for k, v in state.items() if not k.startswith("detr.class_embed")}
print(model.load_state_dict(state, strict=False))
model.cuda()
for name, p in model.named_parameters():
    p.requires_grad = "class_embed" in name or "bbox_embed" in name
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
loader = torch.utils.data.DataLoader(Tiny(), batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=2, drop_last=True)
log = []
for epoch in range(EPOCHS):
    stats = train_one_epoch(model, criterion, loader, optimizer, torch.device("cuda"), epoch, 0.1)
    log.append({k: stats[k] for k in ("loss", "loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice", "class_error")})
    print("EPOCH", epoch, json.dumps(log[-1]), flush=True)
json.dump(log, open("outputs/overfit_smoke.json", "w"), indent=1)
print("peak VRAM GB", torch.cuda.max_memory_allocated() / 2**30)
