"""Build the repo's model with the official COCO panoptic checkpoint loaded strictly."""
import types

import torch

from models.DETR import build_model


def coco_panoptic_args(device):
    return types.SimpleNamespace(
        device=device, position_embedding="sine", backbone="resnet50", dilation=False,
        lr_backbone=0, masks=True, hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048,
        enc_layers=6, dec_layers=6, pre_norm=False, num_queries=100, aux_loss=True,
        set_cost_class=1, set_cost_bbox=5, set_cost_giou=2, frozen_weights=None,
        bbox_loss_coef=5, mask_loss_coef=1, dice_loss_coef=1, giou_loss_coef=2, eos_coef=0.1,
        dataset_file="coco_panoptic", num_classes=250,
        is_thing_map={i: i <= 90 for i in range(201)},
    )


def load_pretrained(checkpoint, device="cuda"):
    args = coco_panoptic_args(device)
    model, criterion, postprocessors = build_model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device), criterion, postprocessors, args
