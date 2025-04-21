import torch
from layoutdit.loss import compute_loss, assign_targets_to_patches

def test_assign_targets_to_patches_single_patch():
    # Setup a 1x1 patch grid for a 100x100 image
    H, W = 100, 100
    patch_rows, patch_cols = 1, 1

    # One ground-truth box in the image
    gt_boxes = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    gt_labels = torch.tensor([3], dtype=torch.long)

    target_classes, target_boxes = assign_targets_to_patches(
        gt_boxes, gt_labels, (patch_rows, patch_cols), (H, W)
    )

    # Should assign the label to the only patch and normalize the box
    assert target_classes.shape == (1,)
    assert int(target_classes[0]) == 3
    expected = torch.tensor([10.0/W, 20.0/W, 30.0/W, 40.0/W])
    assert torch.allclose(target_boxes[0], expected, atol=1e-6)

def test_compute_loss_perfect_prediction():
    # One image, one patch, two classes (0=background,1=foreground)
    # Make the model extremely confident on class 1
    class_logits = torch.tensor([[[-100.0, 100.0]]], requires_grad=True)  # [B=1, P=1, C=2]
    # Predict the exact normalized box [0.5,0.5,0.5,0.5]
    bbox_preds = torch.full((1, 1, 4), 0.5, requires_grad=True)
    # Dummy image tensor
    batch_imgs = torch.zeros((1, 3, 100, 100))
    # Ground-truth box centered in the image
    targets = [{
        "boxes": torch.tensor([[25.0, 25.0, 75.0, 75.0]]),
        "labels": torch.tensor([1], dtype=torch.long)
    }]

    loss = compute_loss(class_logits, bbox_preds, targets, batch_imgs, device="cpu")

    # With perfect classification and regression, loss should be nearly zero
    assert loss.item() < 1e-3

    # Check that gradients can flow
    loss.backward()
    assert class_logits.grad is not None
    assert bbox_preds.grad is not None
