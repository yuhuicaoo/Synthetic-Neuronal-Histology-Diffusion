from monai.networks.nets import HoVerNet
import torch
import os




def create_hovernet_model(stage, out_classes, pretrained_model, ckpt_path, device):
    freeze = (stage == 0)
    model = HoVerNet(
        mode = "fast",
        in_channels = 3,
        out_classes = out_classes,
        act = ("relu", {"inplace": True}),
        norm = "batch",
        pretrained_url = pretrained_model if stage == 0 else None,
        freeze_encoder = freeze
    )
    if stage != 0:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    return model.to(device)