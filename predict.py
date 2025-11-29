# predict.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import cv2 

from model import SimplifiedBIT
from dataset import LevirCdDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose((1,2,0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def refine_mask(mask):
    """
    Refines the binary mask using morphological operations.
    """
    mask = (mask * 255).astype(np.uint8)
    
    # 1. Morphological Cleanup (Removes noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2. Polygon Approximation (Squaring)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_mask = np.zeros_like(mask)
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(refined_mask, [approx], 0, 255, -1)
    
    return (refined_mask > 0).astype(np.uint8)

def predict(args):
    print("Using device:", DEVICE)
    
    model = SimplifiedBIT(num_classes=2).to(DEVICE)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    test_dataset = LevirCdDataset(root_dir=args.data, split="test", img_size=args.img_size)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    os.makedirs(args.output, exist_ok=True)

    print(f"Running prediction (TTA + Refinement) on {args.num_samples} samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples: break
            
            img_a = batch['image_a'].to(DEVICE)
            img_b = batch['image_b'].to(DEVICE)
            mask  = batch['mask'].squeeze(0).cpu().numpy()

            # --- START TTA (Test-Time Augmentation) ---
            
            # 1. Standard Prediction
            pred_standard = model(img_a, img_b)

            # 2. Horizontal Flip
            a_flip = torch.flip(img_a, dims=[3])
            b_flip = torch.flip(img_b, dims=[3])
            out_flip = model(a_flip, b_flip)
            pred_flip = torch.flip(out_flip, dims=[3])

            # 3. Rotation (90 degrees)
            a_rot = torch.rot90(img_a, k=1, dims=[2, 3])
            b_rot = torch.rot90(img_b, k=1, dims=[2, 3])
            out_rot = model(a_rot, b_rot)
            pred_rot = torch.rot90(out_rot, k=-1, dims=[2, 3])

            # Average the predictions
            final_logits = (pred_standard + pred_flip + pred_rot) / 3.0
            
            # --- END TTA ---

            # Get raw binary mask
            pred_raw = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()
            
            # --- APPLY POST-PROCESSING ---
            pred_refined = refine_mask(pred_raw)

            # Visualization
            a = denormalize(img_a.squeeze(0))
            b = denormalize(img_b.squeeze(0))

            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            axes[0].imshow(a); axes[0].set_title("A")
            axes[1].imshow(b); axes[1].set_title("B")
            axes[2].imshow(mask, cmap='gray'); axes[2].set_title("GT")
            axes[3].imshow(pred_refined, cmap='gray'); axes[3].set_title("Pred")
            for ax in axes: ax.axis('off')

            out_path = os.path.join(args.output, f"pred_{i}.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close(fig)
            print("Saved:", out_path)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/23ucs622/BIT_CD/LEVIR-CD-256")
    parser.add_argument("--model_path", type=str, default="/home/23ucs622/BIT_CD/checkpoints/best_bit.pth")
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()
    predict(args)