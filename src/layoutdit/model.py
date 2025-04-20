import datetime
from typing import List, Dict, Optional

import fsspec
import torch
from torch import Tensor
from transformers import AutoModel, AutoConfig
import torch.nn as nn

from layoutdit.log import get_logger

logger = get_logger(__name__)


class LayoutDetectionModel(nn.Module):
    def __init__(
        self,
        num_classes=5,
        previous_layout_dit_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            num_classes (int): number of classes in the dataset (not including background)
        """
        super().__init__()

        self.checkpoint_bucket = "layoutdit"
        self.checkpoint_date = str(datetime.datetime.now())

        hf_backbone_name = "microsoft/dit-base"

        if previous_layout_dit_checkpoint:
            # only fetch the architecture, skip pretrained weights download since we will load the state dict
            logger.info(f"Building architecture from config for '{hf_backbone_name}'")
            config = AutoConfig.from_pretrained(hf_backbone_name)
            self.backbone = AutoModel.from_config(config)
        else:
            # pull full pretrained DiT-base weights
            logger.info(f"Loading pretrained backbone '{hf_backbone_name}'")
            self.backbone = AutoModel.from_pretrained(hf_backbone_name)

        hidden_dim = self.backbone.config.hidden_size
        # classification head outputs num_classes+1 (as we account for background)
        self.classifier = nn.Linear(hidden_dim, num_classes + 1)
        # regress head: outputs 4 coordinates for the corners of the bboxes
        self.bbox_regressor = nn.Linear(hidden_dim, 4)

        if previous_layout_dit_checkpoint:
            assert device, "Please supply a device"

            logger.info(
                f"Loading checkpoint from GCS: {previous_layout_dit_checkpoint}"
            )
            fs = fsspec.filesystem("gcs")
            with fs.open(previous_layout_dit_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=device)
            self.load_state_dict(state_dict)
        else:
            logger.info("No GCS checkpoint provided; using current weights")

    def forward(self, images):
        # images: a batch of images as tensors, shape [B, 3, H, W]
        outputs = self.backbone(images)  # forward pass through transformer

        # The backbone (AutoModel) returns a BaseModelOutput with 'last_hidden_state'
        if hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state  # shape [B, num_tokens, hidden_dim]
        else:
            feats = outputs[0]  # in case it returns tuple

        # drop the [CLS] token, now shape [B, P, hidden_dim] where P = number of patches
        patch_feats = feats[:, 1:, :]

        # Class logits and bbox deltas for each patch
        class_logits = self.classifier(patch_feats)  # [B, P, num_classes+1]
        raw_bbox = self.bbox_regressor(patch_feats)  # [B, P, 4]

        # normalizing here to bbox preds are normalized 0,1 as that is what loss computation expects,
        # see assign_targets_to_patches in loss.py
        bbox_preds = torch.sigmoid(raw_bbox)

        return class_logits, bbox_preds

    @torch.no_grad()
    def predict(self, images: Tensor, targets: List[Dict], score_thresh: float = 0.05):
        class_logits, bbox_preds = self(images)
        num_batches, num_patches, num_classes = class_logits.shape
        H, W = images.shape[-2:]
        grid = int(num_patches**0.5)
        patch_h, patch_w = H / grid, W / grid

        scores, labels = torch.softmax(class_logits, -1).max(-1)
        all_results = []

        for batch in range(num_batches):
            img_id = int(targets[batch]["image_id"])
            H0, W0 = targets[batch]["orig_size"]
            batch_scores = scores[batch]
            batch_labels = labels[batch]

            # remember that model predicts at patch level dx, dy, dw, dh
            pred_bbox_patches = bbox_preds[batch]

            for patch in range(num_patches):
                score = float(batch_scores[patch])
                label = int(batch_labels[patch])
                if label == 0 or score < score_thresh:
                    continue

                dx, dy, dw, dh = pred_bbox_patches[patch].tolist()
                row, col = divmod(patch, grid)

                # 1) center in processed image
                cx_p = (col + dx) * patch_w
                cy_p = (row + dy) * patch_h

                # 2) box size in 224 space
                w_p = dw * W
                h_p = dh * H

                # 3) corner coords in 224 space
                x1 = cx_p - w_p / 2
                y1 = cy_p - h_p / 2
                x2 = x1 + w_p
                y2 = y1 + h_p

                # 4) clamp to account for numerical rounding
                x1 = max(0.0, min(x1, W))
                y1 = max(0.0, min(y1, H))
                x2 = max(0.0, min(x2, W))
                y2 = max(0.0, min(y2, H))

                # 5) compute w_p,h_p after clamp
                w = x2 - x1
                h = y2 - y1

                # 6) move back ro original image space
                x1 = x1 * (W0 / W)
                y1 = y1 * (H0 / H)
                w = w * (W0 / W)
                h = h * (H0 / H)

                all_results.append(
                    {
                        "image_id": img_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "score": score,
                    }
                )

        return all_results

    def save_checkpoint_to_gcs(self, epoch: int) -> str:
        """
        Save the model's state_dict to the class’s DEFAULT_BUCKET on GCS:
          gs://{DEFAULT_BUCKET}/model_checkpoints/{run_date}/epoch_{epoch}.pth
        Returns the GCS path.
        """
        model_device = next(self.parameters()).device
        fs = fsspec.filesystem("gcs")
        base_path = (
            f"gs://{self.checkpoint_bucket}/"
            f"model_checkpoints/{self.checkpoint_date}/epoch_{epoch}"
        )

        paths: Dict[str, str] = {}

        # 1) CPU‐safe checkpoint
        cpu_path = base_path + "_cpu.pth"
        with fs.open(cpu_path, "wb") as f:
            logger.debug(f"Saving CPU checkpoint to {cpu_path}")
            # copy every tensor to CPU
            cpu_state = {k: v.cpu() for k, v in self.state_dict().items()}
            torch.save(cpu_state, f)
        paths["cpu"] = cpu_path

        # 2) GPU checkpoint (only if model is actually on CUDA)
        if model_device.type == "cuda":
            gpu_path = base_path + "_gpu.pth"
            with fs.open(gpu_path, "wb") as f:
                logger.debug(f"Saving GPU checkpoint to {gpu_path}")
                # state_dict() already contains cuda tensors
                torch.save(self.state_dict(), f)
            paths["cuda"] = gpu_path

        return paths
