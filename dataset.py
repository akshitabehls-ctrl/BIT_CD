import os
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class LevirCdDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.a_dir = os.path.join(root_dir, split, "A")
        self.b_dir = os.path.join(root_dir, split, "B")
        self.m_dir = os.path.join(root_dir, split, "label")

        try:
            self.files = sorted([
                f for f in os.listdir(self.a_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ])
        except FileNotFoundError:
            self.files = []
            print(f"Warning!!: Could not find data at {self.a_dir}")

        # Normalization transform
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform(self, img_a, img_b, mask):
        #Applies complex augmentations for training,but only resizing/normalization for validation.
        # 1. Resize (Always)
        img_a = TF.resize(img_a, (self.img_size, self.img_size), interpolation=Image.BICUBIC)
        img_b = TF.resize(img_b, (self.img_size, self.img_size), interpolation=Image.BICUBIC)
        mask  = TF.resize(mask,  (self.img_size, self.img_size), interpolation=Image.NEAREST)

        # 2. Augmentations (ONLY for Training)
        if self.split == "train":
            
            # --- A. Geometric (Must be consistent across A, B, Mask) ---
            
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
                mask  = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                img_a = TF.vflip(img_a)
                img_b = TF.vflip(img_b)
                mask  = TF.vflip(mask)

            # Random Rotation (90, 180, 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img_a = TF.rotate(img_a, angle)
                img_b = TF.rotate(img_b, angle)
                mask  = TF.rotate(mask,  angle)

            # --- B. Photometric (Independent for A and B) ---
            # This solves the "Day/Night" and "Complex Env" issues
            
            # Define the color jitter transform (Brightness, Contrast, Saturation, Hue)
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            
            # Apply randomly to A
            if random.random() > 0.2:
                img_a = color_jitter(img_a)
            
            # Apply randomly to B (Independent of A!)
            if random.random() > 0.2:
                img_b = color_jitter(img_b)

            # --- C. Blur & Noise (The Professor's "Deblur/Noise" Note) ---
            
            # Gaussian Blur (Randomly apply to A or B)
            if random.random() > 0.3:
                sigma = random.uniform(0.1, 2.0)
                img_a = img_a.filter(ImageFilter.GaussianBlur(sigma))
            
            if random.random() > 0.3:
                sigma = random.uniform(0.1, 2.0)
                img_b = img_b.filter(ImageFilter.GaussianBlur(sigma))

        # 3. ToTensor & Normalize
        img_a = self.normalize(img_a)
        img_b = self.normalize(img_b)
        
        # Mask to Tensor (0 or 1)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        return img_a, img_b, mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load images
        img_a = Image.open(os.path.join(self.a_dir, fname)).convert("RGB")
        img_b = Image.open(os.path.join(self.b_dir, fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.m_dir, fname)).convert("L")

        # Apply the smart transform pipeline
        img_a, img_b, mask = self.transform(img_a, img_b, mask)

        return {
            "image_a": img_a,
            "image_b": img_b,
            "mask": mask
        }