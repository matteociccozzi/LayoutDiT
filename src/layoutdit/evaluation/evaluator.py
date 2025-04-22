import json
from typing import Callable

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torch.utils.data import DataLoader
import os
import fsspec
from PIL import Image, ImageDraw, ImageFont

from layoutdit.configuration import LayoutDitConfig
from layoutdit.configuration.config_constructs import DataLoaderConfig
from layoutdit.data.publay_dataset import PubLayNetDataset, collate_fn
from layoutdit.data.transforms import layout_dit_transforms
from layoutdit.log import get_logger
from layoutdit.model import LayoutDetectionModel

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, model: LayoutDetectionModel, layout_dit_config: LayoutDitConfig):
        self.eval_config = layout_dit_config.eval_config

        self.fs_open: Callable = fsspec.open

        self.model = model.to(self.eval_config.device).eval()
        self.dataloader = self._build_eval_dataloader(
            layout_dit_config.data_loader_config, layout_dit_config.local_mode
        )
        self.device = self.eval_config.device

        self.coco_gt = self._get_coco_gt(dataloader=self.dataloader)

        # used in visualizations
        self.id2cat_map = {
            cat["id"]: cat.get("name", str(cat["id"]))
            for cat in self.coco_gt.dataset["categories"]
        }

        self.score_thresh = self.eval_config.score_thresh

        logger.debug("Successfully initialized evaluator")

    @staticmethod
    def _get_coco_gt(dataloader: DataLoader):
        ds = dataloader.dataset

        # dataset impl will load/save this, avoid reloading the coco annotations as files might be large
        if hasattr(ds, "coco_data"):
            coco = COCO()
            coco.dataset = ds.coco_data
            coco.createIndex()

            return coco
        raise ValueError("Please supply the the coco_data attribute in the Dataset")

    def visualize_preds(self):
        """
        Reads predictions from self.eval_config.predictions_path,
        draws boxes on up to self.eval_config.num_images source images,
        and writes them into self.eval_config.visualize_dir.
        """
        # 1) Grab config values
        preds_path = self.eval_config.predictions_path
        max_per_img = self.eval_config.max_per_image
        out_dir = self.eval_config.visualize_dirpath_prefix + "_preds"
        num_images = self.eval_config.num_images
        img_root = self.dataloader.dataset.images_root_dir

        os.makedirs(out_dir, exist_ok=True)

        # 2) Load predictions.json
        with self.fs_open(preds_path, "r") as f:
            all_preds = json.load(f)

        # 3) Group predictions by image_id
        preds_by_image = {}
        for p in all_preds:
            preds_by_image.setdefault(p["image_id"], []).append(p)

        # 4) Prepare font
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()

        # 5) Iterate through COCO images, stop after num_images
        count = 0
        for img_rec in self.coco_gt.dataset["images"]:
            img_id = img_rec["id"]
            if img_id not in preds_by_image:
                continue

            if num_images is not None and count >= num_images:
                break
            count += 1

            # -- load the raw image --
            file_name = img_rec["file_name"]
            full_path = os.path.join(img_root, file_name)
            fs = fsspec.open(full_path, mode="rb").fs
            with fs.open(full_path, "rb") as f:
                img = Image.open(f).convert("RGB")

            draw = ImageDraw.Draw(img)
            W, H = img.size

            # -- pick top-k preds for this image --
            preds = sorted(
                preds_by_image[img_id], key=lambda x: x["score"], reverse=True
            )[:max_per_img]

            # -- draw each box + label --
            for p in preds:
                x, y, w, h = p["bbox"]
                cat, _ = p["category_id"], p.get("score")

                x0 = max(0, min(x, W))
                y0 = max(0, min(y, H))
                x1 = max(0, min(x + w, W))
                y1 = max(0, min(y + h, H))

                # draw the bounding box
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

                # build the label text
                label = self.id2cat_map.get(
                    cat, str(cat)
                )  # + (f" {score:.2f}" if score is not None else "")

                # === Updated text size computation ===
                # Pillow 10+ removed draw.textsize(); use textbbox() instead
                left, top, right, bottom = draw.textbbox(
                    (x0, y0), label, font=font
                )  # :contentReference[oaicite:2]{index=2}
                tw = right - left
                th = bottom - top

                # draw background rectangle for the label
                draw.rectangle([x0, y0 - th, x0 + tw, y0], fill="red")
                # draw the label text
                draw.text((x0, y0 - th), label, fill="white", font=font)

            # -- save out the visualization --
            out_path = os.path.join(out_dir, f"{img_id}.jpg")
            img.save(out_path)
            logger.info(f"Saved visualization for image {img_id} to {out_path}")

    def visualize_gt(self):
        """
        Draws ground‑truth boxes on up to self.eval_config.num_images
        source images and writes them into self.eval_config.visualize_dir.
        """
        # 1) Grab config values
        out_dir = self.eval_config.visualize_dirpath_prefix + "_gt"
        num_images = self.eval_config.num_images
        img_root = self.dataloader.dataset.images_root_dir

        os.makedirs(out_dir, exist_ok=True)

        # 2) Group GT annotations by image_id
        gt_by_image = {}
        for ann in self.coco_gt.dataset["annotations"]:
            img_id = ann["image_id"]
            gt_by_image.setdefault(img_id, []).append(ann)

        # 4) Prepare font
        try:
            font = ImageFont.truetype("arial.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()

        # 5) Iterate through COCO images, stop after num_images
        count = 0
        for img_rec in self.coco_gt.dataset["images"]:
            img_id = img_rec["id"]
            if img_id not in gt_by_image:
                continue

            if num_images is not None and count >= num_images:
                break
            count += 1

            # -- load the raw image --
            file_name = img_rec["file_name"]
            full_path = os.path.join(img_root, file_name)
            fs = fsspec.open(full_path, mode="rb").fs
            with fs.open(full_path, "rb") as f:
                img = Image.open(f).convert("RGB")

            draw = ImageDraw.Draw(img)
            W, H = img.size

            # -- draw each GT box + label --
            for ann in gt_by_image[img_id]:
                x, y, w, h = ann["bbox"]
                cat_id = ann["category_id"]
                label_text = self.id2cat_map.get(cat_id, str(cat_id))

                # clamp coords to image
                x0 = max(0, min(x, W))
                y0 = max(0, min(y, H))
                x1 = max(0, min(x + w, W))
                y1 = max(0, min(y + h, H))

                # draw the box in green for GT
                draw.rectangle([x0, y0, x1, y1], outline="green", width=2)

                # measure text size
                left, top, right, bottom = draw.textbbox(
                    (x0, y0), label_text, font=font
                )
                tw = right - left
                th = bottom - top

                # draw label background & text
                draw.rectangle([x0, y0 - th, x0 + tw, y0], fill="green")
                draw.text((x0, y0 - th), label_text, fill="white", font=font)

            # -- save visualization --
            out_path = os.path.join(out_dir, f"{img_id}_gt.jpg")
            img.save(out_path)
            logger.info(f"Saved GT visualization for image {img_id} to {out_path}")

    def score(self):
        all_predictions = []

        self.model.eval()
        with torch.no_grad():
            for images, targets in self.dataloader:
                # images: tuple of PIL Images after transforms
                # targets: list of dicts with keys 'boxes','labels','image_id','orig_size'

                # Move images to device and call the model
                imgs_t = [img.to(self.device) for img in images]
                batch_outputs = self.model(imgs_t)
                # batch_outputs is a list of dicts, one per image, with keys:
                #   'boxes'   Tensor[K,4] in [x1,y1,x2,y2] (on the 224×224 or original scale)
                #   'labels'  Tensor[K]
                #   'scores'  Tensor[K]

                # 2) Convert each image’s detections into COCO‐style JSON entries
                for tgt, out in zip(targets, batch_outputs):
                    # Determine the image_id (could be a tensor)
                    img_id = tgt["image_id"].item() if isinstance(tgt["image_id"], torch.Tensor) else tgt["image_id"]

                    boxes = out["boxes"].cpu()
                    labels = out["labels"].cpu()
                    scores = out["scores"].cpu()


                    for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box.tolist()
                        all_predictions.append({
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                        })

        # 3) If we got any predictions, write them out and run COCOeval
        if not all_predictions:
            logger.warning("No predictions were generated.")
            return None

        # Save to JSON
        self.save_preds_json(all_predictions)

        # Load into pycocotools and evaluate
        coco_dt = self.coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Map the 12 COCO stats to your keys
        coco_keys = [
            "mAP", "AP50", "AP75",
            "AP_s", "AP_m", "AP_l",
            "AR1", "AR10", "AR100",
            "AR_s", "AR_m", "AR_l"
        ]
        return dict(zip(coco_keys, coco_eval.stats.tolist()))

    def save_preds_json(self, all_predictions):
        if self.eval_config.predictions_path:
            with self.fs_open(self.eval_config.predictions_path, "w") as f:
                json.dump(all_predictions, f)

            logger.info(
                f"Saved {len(all_predictions)} predictions to {self.eval_config.predictions_path}"
            )

    @staticmethod
    def _build_eval_dataloader(
        dataloader_config: DataLoaderConfig, local_mode: bool
    ) -> DataLoader:
        if local_mode:
            data_segment = "single"
        else:
            data_segment = "val"

        dataset = PubLayNetDataset(
            images_root_dir=f"gs://layoutdit/data/{data_segment}/",
            annotations_json_path=f"gs://layoutdit/data/{data_segment}.json"
        )

        return DataLoader(
            dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=dataloader_config.shuffle,
            num_workers=dataloader_config.num_workers,
            collate_fn=collate_fn,
        )
