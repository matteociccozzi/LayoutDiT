from transformers import AutoModel
import torch.nn as nn


class LayoutDetectionModel(nn.Module):
    def __init__(self, num_classes=5):
        """
        Args:
            num_classes (int): number of classes in the dataset (not including background)
        """
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained("microsoft/dit-base")

        hidden_dim = self.backbone.config.hidden_size
        # classification head outputs num_classes+1 (as we account for background)
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)
        # regress head: outputs 4 coordinates for the corners of the bboxes
        self.bbox_regressor = nn.Linear(hidden_dim, 4)

    def forward(self, images):
        # images: a batch of images as tensors, shape [B, 3, H, W]
        # Use the processor to get pixel_values (with resizing/normalization if needed)
        # For simplicity, assume images are already normalized and of correct size
        # If using AutoImageProcessor, do: pixel_values = processor(images=list_of_PIL_images, return_tensors='pt').pixel_values
        # Ensure input device matches model device
        outputs = self.backbone(images)  # forward pass through transformer
        # The backbone (AutoModel) returns a BaseModelOutput with 'last_hidden_state'
        if hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state  # shape [B, num_tokens, hidden_dim]
        else:
            feats = outputs[0]  # in case it returns tuple

        # Typically, DiT (like ViT) has a CLS token at index 0. We can ignore it for detection.
        patch_feats = feats[
            :, 1:, :
        ]  # drop the [CLS] token, now shape [B, P, hidden_dim] where P = number of patches
        # Class logits and bbox deltas for each patch
        class_logits = self.classifier(patch_feats)  # [B, P, num_classes+1]
        bbox_preds = self.bbox_regressor(patch_feats)  # [B, P, 4]

        return class_logits, bbox_preds
