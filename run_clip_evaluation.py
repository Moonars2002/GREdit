import argparse
import os
import re

import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image):  
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(self, image_0, image_1, text_0, text_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image


class ImagePairDataset(Dataset):
    def __init__(self, paths0, paths1):
        self.paths0 = paths0
        self.paths1 = paths1
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.paths0)

    def __getitem__(self, idx):
        path0 = self.paths0[idx]
        path1 = self.paths1[idx]
        
        img0 = Image.open(path0).convert("RGB")
        img1 = Image.open(path1).convert("RGB")
        
        return self.transform(img0), self.transform(img1)


def get_sorted_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    
    def natural_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

    files.sort(key=natural_key)
    return [os.path.join(folder, f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP similarities with Fixed View Selection.")
    parser.add_argument("--image_dir0", required=True, help="Dir of original images")
    parser.add_argument("--image_dir1", required=True, help="Dir of edited images")
    parser.add_argument("--text0", default="a photograph", help="Description for original images.")
    parser.add_argument("--text1", default="Turn him into a spider man with Mask", help="Target description.")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model name.")
    parser.add_argument("--device", default=None, help="device: 'cpu' or 'cuda'.")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="CPU workers.")
    
    parser.add_argument("--interval", type=int, default=8, help="Sampling interval. Default is 8 (every 8th image) for EditSplat protocol.")
    
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Batch Size: {args.batch_size}")

    paths0 = get_sorted_image_files(args.image_dir0)
    paths1 = get_sorted_image_files(args.image_dir1)

    if len(paths0) == 0 or len(paths1) == 0:
        raise ValueError("No images found.")
    if len(paths0) != len(paths1):
        min_len = min(len(paths0), len(paths1))
        print(f"Warning: Mismatch {len(paths0)} vs {len(paths1)}. Truncating to {min_len}.")
        paths0 = paths0[:min_len]
        paths1 = paths1[:min_len]

    if args.interval > 1:
        print(f"Applying fixed selection: Taking every {args.interval}th image.")
        paths0 = paths0[::args.interval]
        paths1 = paths1[::args.interval]
    
    n_all = len(paths0)
    print(f"Final evaluation set size: {n_all} pairs.")

    dataset = ImagePairDataset(paths0, paths1)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = ClipSimilarity(name=args.model)
    model.to(device)
    model.eval()

    with torch.no_grad():
        target_text_feat = model.encode_text([args.text1])
        src_text_feat = model.encode_text([args.text0])

    all_sims_1 = []
    all_sims_dir = []
    all_sims_img = []

    print("Start evaluating...")

    with torch.no_grad():
        for batch_img0, batch_img1 in tqdm(dataloader, desc="Processing Batches"):
            batch_img0 = batch_img0.to(device)
            batch_img1 = batch_img1.to(device)
            
            img_feat0 = model.encode_image(batch_img0)
            img_feat1 = model.encode_image(batch_img1)

            sim_1 = F.cosine_similarity(img_feat1, target_text_feat)
            
            img_diff = img_feat1 - img_feat0
            text_diff = target_text_feat - src_text_feat

            sim_direction = F.cosine_similarity(img_diff, text_diff)

            sim_image = F.cosine_similarity(img_feat0, img_feat1)

            all_sims_1.append(sim_1.cpu())
            all_sims_dir.append(sim_direction.cpu())
            all_sims_img.append(sim_image.cpu())

    final_sim_1 = torch.cat(all_sims_1).mean().item()
    final_sim_dir = torch.cat(all_sims_dir).mean().item()
    final_sim_img = torch.cat(all_sims_img).mean().item()

    print(f"\nResults (Mean over {n_all} selected views):")
    print(f"  [Text-Image Consistency] (sim_1):   {final_sim_1:.4f}")
    print(f"  [Edit Quality] (sim_dir):           {final_sim_dir:.4f}")
    print(f"  [Image Identity] (sim_img):         {final_sim_img:.4f}")

if __name__ == "__main__":
    main()
