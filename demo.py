from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from copy import deepcopy

device = torch.device('mps')

artifact = wandb.use_artifact(used_artifact)
artifact_dir = artifact.download()
checkpoint = torch.load(artifact_dir+"/checkpoint.pth", map_location=torch.device('mps'))

model, criterion, postprocessors = build_model(args)
model.load_state_dict(checkpoint['model'])
model.eval();

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

im = Image.open("sample.png")
img = transform(im).unsqueeze(0)
# print the resolition of the image
print(img.shape)

out = model(img)
result = postprocessors['panoptic'](out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

import matplotlib.pyplot as plt
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(out["pred_masks"][keep]):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask.detach().numpy(), cmap="cividis")
    ax.axis('off')
fig.tight_layout()

# We extract the segments info and the panoptic result from DETR's prediction
segments_info = deepcopy(result["segments_info"])
# Panoptic predictions are stored in a special format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
final_w, final_h = panoptic_seg.size
# We convert the png into an segment id map
panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

meta = MetadataCatalog.get("cityscapes_fine_panoptic_val")
for i in range(len(segments_info)):
    c = segments_info[i]["category_id"]
    segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

# Finally we visualize the prediction
v = Visualizer(np.array(im.copy().resize((final_w, final_h)))[:, :, ::-1], meta, scale=1.0)
v._default_font_size = 20
v = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info, area_threshold=0)
# cv2_imshow(v.get_image())
# im

# display the image
plt.imshow(v.get_image()[:, :, ::-1])