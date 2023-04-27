import random
import numpy as np
import torch

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)     # missing some deterministic impl

device = torch.device("cuda:0")

class Args:
    pass
args = Args()
# Postitional encoding
args.position_embedding = "sine"
# CNN Backbone
args.backbone = "resnet50"
args.dilation = None
# Hungarian matcher
args.set_cost_class = 1
args.set_cost_bbox = 5
args.set_cost_giou = 2
# Transformer
args.hidden_dim = 256
args.dropout = 0.1
args.nheads = 8
args.dim_feedforward = 2048
args.enc_layers = 6
args.dec_layers = 6
args.pre_norm = None
# DETR
args.num_queries = 100
args.aux_loss = True # calculate loss at eache decoder layer
args.masks = True
args.frozen_weights = None
args.bbox_loss_coef = 5
args.mask_loss_coef = 1
args.dice_loss_coef = 1
args.giou_loss_coef = 2
args.eos_coef = 0.1
# Dataset
args.dataset_file = "coco_panoptic" # cityscape
args.coco_path = ""
args.coco_panoptic_path = ""
# Training
args.lr = 1e-4
args.weight_decay = 1e-4
args.lr_backbone = 0    # 0 means frozen backbone
args.batch_size = 1
args.epochs = 300
args.lr_drop = 200
args.clip_max_norm = 0.1

args.output_dir = "out_dir"
args.eval = False

def freeze_attn(model, args):
    for i in range(args.dec_layers):
       for param in model.detr.transformer.decoder.layers[i].self_attn.parameters():
           param.requires_grad = False
       for param in model.detr.transformer.decoder.layers[i].multihead_attn.parameters():
           param.requires_grad = False

    for i in range(args.enc_layers):
        for param in model.detr.transformer.encoder.layers[i].self_attn.parameters():
           param.requires_grad = False

def freeze_decoder(model, args):
    for param in model.detr.transformer.decoder.parameters():
        param.requires_grad = False

def freeze_first_layers(model, args):
    for i in range(args.enc_layers // 2):
        for param in model.detr.transformer.encoder.layers[i].parameters():
            param.requires_grad = False

    for i in range(args.dec_layers // 2):
        for param in model.detr.transformer.decoder.layers[i].parameters():
            param.requires_grad = False


def build_pretrained_model(args):
    pre_trained = torch.hub.load('facebookresearch/detr', 'detr_resnet50_panoptic', pretrained=True, return_postprocessor=False, num_classes=250)
    model, criterion, postprocessors = build_model(args)

    model.detr.backbone.load_state_dict(pre_trained.detr.backbone.state_dict())
    model.detr.bbox_embed.load_state_dict(pre_trained.detr.bbox_embed.state_dict())
    model.detr.query_embed.load_state_dict(pre_trained.detr.query_embed.state_dict())
    model.detr.input_proj.load_state_dict(pre_trained.detr.input_proj.state_dict())
    model.detr.transformer.load_state_dict(pre_trained.detr.transformer.state_dict())
 
    model.bbox_attention.load_state_dict(pre_trained.bbox_attention.state_dict())
    model.mask_head.load_state_dict(pre_trained.mask_head.state_dict())
    
    freeze_attn(model, args)

    return model, criterion, postprocessors

ENABLE_WANDB = True                                      # set if you plan to log on wandb
used_artifact = '2_2_2_transf_unfreeze_aux:latest'            # if set not train from scratch (detre pretrained on COCO)
wandb_experiment_name = "2_2_1_transf_unfreeze_aux"      # set if starting a new run
run_id = None                                            # set to None if starting a new run

if ENABLE_WANDB:
    import wandb
    
    if run_id is not None: 
        wandb.init(project='detr', id=run_id, resume='allow')
    else:
        wandb.init(project='detr', name=wandb_experiment_name)
    
    wandb.config.position_embedding = args.position_embedding
    wandb.config.backbone = args.backbone
    wandb.config.dilation = args.dilation
    wandb.config.set_cost_class = args.set_cost_class
    wandb.config.set_cost_bbox = args.set_cost_bbox
    wandb.config.set_cost_giou = args.set_cost_giou
    wandb.config.hidden_dim = args.hidden_dim
    wandb.config.dropout = args.dropout
    wandb.config.nheads = args.nheads
    wandb.config.dim_feedforward = args.dim_feedforward
    wandb.config.enc_layers = args.enc_layers
    wandb.config.dec_layers = args.dec_layers
    wandb.config.pre_norm = args.pre_norm
    wandb.config.num_queries = args.num_queries
    wandb.config.aux_loss = args.aux_loss
    wandb.config.masks = args.masks
    wandb.config.frozen_weights = args.frozen_weights
    wandb.config.bbox_loss_coef = args.bbox_loss_coef
    wandb.config.mask_loss_coef = args.mask_loss_coef
    wandb.config.dice_loss_coef = args.dice_loss_coef
    wandb.config.giou_loss_coef = args.giou_loss_coef
    wandb.config.eos_coef = args.eos_coef
    wandb.config.lr = args.lr
    wandb.config.weight_decay = args.weight_decay
    wandb.config.lr_backbone = args.lr_backbone
    wandb.config.batch_size = args.batch_size
    wandb.config.epochs = args.epochs
    wandb.config.lr_drop = args.lr_drop
    wandb.config.clip_max_norm = args.clip_max_norm