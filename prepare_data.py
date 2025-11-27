# prepare_data.py
import os
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
# This is the folder you just renamed (the source 1024x1024 images)
SOURCE_ROOT = "LEVIR-CD-HUGE"  

# This is the NEW folder we will create (the 256x256 patches)
TARGET_ROOT = "LEVIR-CD-256"     
SPLITS = ['train', 'val', 'test']
CROP_SIZE = 256
# ---------------------

def crop_and_save(split):
    print(f"Processing {split}...")
    
    # Source paths
    src_A = os.path.join(SOURCE_ROOT, split, 'A')
    src_B = os.path.join(SOURCE_ROOT, split, 'B')
    src_L = os.path.join(SOURCE_ROOT, split, 'label')
    
    # Target paths
    tgt_A = os.path.join(TARGET_ROOT, split, 'A')
    tgt_B = os.path.join(TARGET_ROOT, split, 'B')
    tgt_L = os.path.join(TARGET_ROOT, split, 'label')
    
    os.makedirs(tgt_A, exist_ok=True)
    os.makedirs(tgt_B, exist_ok=True)
    os.makedirs(tgt_L, exist_ok=True)
    
    # Safety check
    if not os.path.exists(src_A):
        print(f"⚠️ Warning: Could not find {src_A}. Check your folder structure!")
        return

    files = os.listdir(src_A)
    
    count = 0
    for fname in tqdm(files):
        if not (fname.endswith('.png') or fname.endswith('.jpg')): continue
        
        # Load 1024x1024 images
        img_A = Image.open(os.path.join(src_A, fname))
        img_B = Image.open(os.path.join(src_B, fname))
        img_L = Image.open(os.path.join(src_L, fname))
        
        w, h = img_A.size
        
        # If image is already small (256x256), just copy it
        if w == CROP_SIZE and h == CROP_SIZE:
            img_A.save(os.path.join(tgt_A, fname))
            img_B.save(os.path.join(tgt_B, fname))
            img_L.save(os.path.join(tgt_L, fname))
            count += 1
            continue

        # Otherwise, crop it!
        for r in range(0, h, CROP_SIZE):
            for c in range(0, w, CROP_SIZE):
                box = (c, r, c + CROP_SIZE, r + CROP_SIZE)
                
                patch_A = img_A.crop(box)
                patch_B = img_B.crop(box)
                patch_L = img_L.crop(box)
                
                new_name = f"{os.path.splitext(fname)[0]}_{r}_{c}.png"
                
                patch_A.save(os.path.join(tgt_A, new_name))
                patch_B.save(os.path.join(tgt_B, new_name))
                patch_L.save(os.path.join(tgt_L, new_name))
                count += 1
    
    print(f"Created {count} patches for {split}.")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ Error: Could not find folder '{SOURCE_ROOT}'")
        print("Did you rename your uploaded folder to 'LEVIR-CD-HUGE'?")
    else:
        print("✂️  Starting crop process...")
        for s in SPLITS:
            crop_and_save(s)
        print("\n✅ Done! Your new dataset is in 'LEVIR-CD-256'")