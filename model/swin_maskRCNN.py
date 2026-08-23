import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from transformers import AutoModel
from collections import OrderedDict
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone that extracts features at multiple stages.
    """
    def __init__(self, model_name="microsoft/swinv2-small-patch4-window16-256", cache_dir='/eresearch/hie/ycao891/Yuhui/hf_cache'):
        super().__init__()
        # Load pretrained Swin model
        self.swin = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Based on actual outputs:
        # hidden_state 0: (1, 4096, 96) -> 64x64 feature map
        # hidden_state 1: (1, 1024, 192) -> 32x32 feature map  
        # hidden_state 2: (1, 256, 384) -> 16x16 feature map
        # hidden_state 3: (1, 64, 768) -> 8x8 feature map
        self.out_channels = [96, 192, 384, 768]
        
        # Create dummy modules for each stage so IntermediateLayerGetter can find them
        # self.stage0 = nn.Identity()
        # self.stage1 = nn.Identity()
        # self.stage2 = nn.Identity()
        # self.stage3 = nn.Identity()
        
        # # Store features during forward pass
        # self._features = {}
        
    def forward(self, x):
        # Get outputs from all stages
        outputs = self.swin(x, output_hidden_states=True, return_dict=True)
        
        # Extract features from different stages
        hidden_states = outputs.hidden_states
        
        # Process each stage
        features = OrderedDict()
        for i in range(4):
            # Reshape from (B, L, C) to (B, C, H, W)
            feat = hidden_states[i]
            B, L, C = feat.shape
            H = W = int(L ** 0.5)
            feat = feat.transpose(1, 2).reshape(B, C, H, W)
            features[str(i)] = feat
        
        # # Pass through identity modules (for IntermediateLayerGetter compatibility)
        # x0 = self.stage0(features[0])
        # x1 = self.stage1(features[1])
        # x2 = self.stage2(features[2])
        # x3 = self.stage3(features[3])
        
        # Return the last one (but intermediate layer getter will capture all)
        return features


class SwinWithFPN(nn.Module):
    """
    Swin backbone with FPN on top for feature pyramid.
    """
    def __init__(self, model_name="microsoft/swinv2-small-patch4-window16-256", 
                 cache_dir='/eresearch/hie/ycao891/Yuhui/hf_cache'):
        super().__init__()
        
        # Create Swin backbone
        backbone = SwinBackbone(model_name, cache_dir)
        
        # Channels for all 4 stages
        in_channels_list = [96, 192, 384, 768]
        out_channels = 256  # FPN output channels
        
        # We need to manually handle the feature extraction
        self.backbone = backbone
        self.out_channels = out_channels
        
        # Create FPN with LastLevelMaxPool
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        
    def forward(self, x):
        # Get features from Swin at all stages
        # outputs = self.backbone.swin(x, output_hidden_states=True, return_dict=True)
        # hidden_states = outputs.hidden_states
        
        # # Create feature dictionary
        # features = OrderedDict()
        # for i in range(4):
        #     # Reshape from (B, L, C) to (B, C, H, W)
        #     feat = hidden_states[i]
        #     B, L, C = feat.shape
        #     H = W = int(L ** 0.5)
        #     feat = feat.transpose(1, 2).reshape(B, C, H, W)
        #     features[str(i)] = feat
        
        # get features
        features = self.backbone(x)
        # apply FPN
        fpn_features = self.fpn(features)
        return fpn_features


def create_maskrcnn_swin(num_classes=91, 
                         model_name="microsoft/swinv2-small-patch4-window16-256",
                         cache_dir='/eresearch/hie/ycao891/Yuhui/hf_cache',
                         pretrained_backbone=True,
                         min_size=256,
                         max_size=256):
    """
    Creates a Mask R-CNN model with Swin Transformer backbone.
    
    Args:
        num_classes: Number of object classes (including background)
        model_name: HuggingFace model name for Swin
        cache_dir: Cache directory for HuggingFace models
        pretrained_backbone: Whether to use pretrained Swin weights
        min_size: Minimum size for input images
        max_size: Maximum size for input images
    
    Returns:
        MaskRCNN model with Swin backbone
    """
    
    # Create backbone with FPN (no extra_blocks parameter needed anymore)
    backbone = SwinWithFPN(
        model_name=model_name,
        cache_dir=cache_dir
    )

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    # Create Mask R-CNN model
    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size,
        # These are standard Mask R-CNN parameters
        rpn_anchor_generator=None,
        rpn_head=None,  # Will use default
        box_predictor=None,  # Will use default
        box_head=None,  # Will use default
        box_nms_thresh=0.5,   # Default = 0.5,
        box_fg_iou_thresh=0.5, # Default = 0.5
        mask_head=None,  # Will use default
        mask_predictor=None,  # Will use default
        # image_mean=[0.485, 0.456, 0.406],
        # image_std=[0.229, 0.224, 0.225],
    )
    
    return model

def create_maskrcnn_resnet50(num_classes=2, min_size=256, max_size=256):
    
    # Create model with pretrained backbone
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        num_classes=num_classes,
        min_size=min_size,
        max_size=max_size
    )
    
    return model