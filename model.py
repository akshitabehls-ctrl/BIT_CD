import torch
import torch.nn as nn
import torchvision.models as models

# --- Step 1: The CNN Backbone (ResNet50) ---
class FeatureExtractor(nn.Module):
    """
    Refactored ResNet50 to return features from multiple layers (C2, C3, C4).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load the base model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        print("Loaded pretrained ResNet50 backbone")

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # C2: Stride 4, 256 channels
        self.layer2 = resnet.layer2  # C3: Stride 8, 512 channels
        self.layer3 = resnet.layer3  # C4: Stride 16, 1024 channels 
        
        # We do not use layer4 (C5) as it is too deep for this task

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        f_pre = self.maxpool(x)

        f_c2 = self.layer1(f_pre)   # 1/4 res
        f_c3 = self.layer2(f_c2)    # 1/8 res
        f_c4 = self.layer3(f_c3)    # 1/16 res
        
        return f_c2, f_c3, f_c4

# --- Step 2: The Bitemporal Transformer ---
class BIT_Transformer(nn.Module):
    def __init__(self, in_channels=1024, transformer_dim=1024, num_heads=8, num_layers=4):
        super().__init__()
        
        # CORRECTED: 256x256 input -> 1/16 feature map = 16x16
        self.img_size = 16 
        self.num_tokens = self.img_size * self.img_size # 256 tokens
        
        self.input_proj = nn.Conv2d(in_channels, transformer_dim, kernel_size=1)

        self.transformer = nn.Transformer(
            d_model=transformer_dim, 
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=transformer_dim * 4,
            batch_first=False 
        )
        
        self.pos_embed = nn.Parameter(torch.rand(self.num_tokens, 1, transformer_dim))

    def forward(self, features_A, features_B):
        proj_A = self.input_proj(features_A)
        proj_B = self.input_proj(features_B)
        
        # Flatten [B, C, H, W] -> [HW, B, C]
        seq_A = proj_A.flatten(2).permute(2, 0, 1)
        seq_B = proj_B.flatten(2).permute(2, 0, 1)
        
        seq_A = seq_A + self.pos_embed
        seq_B = seq_B + self.pos_embed
        
        transformer_out = self.transformer(src=seq_A, tgt=seq_B)
        
        # Reshape back to [B, C, H, W]
        batch_size = features_A.shape[0]
        out_features = transformer_out.permute(1, 2, 0).view(batch_size, -1, self.img_size, self.img_size)
        
        return out_features

# --- Step 3: The UNet Decoder ---
class UNetDecoder(nn.Module):
    """
    UNet-style decoder for multi-level feature fusion.
    """
    def __init__(self, t_in=1024, f_c4=1024, f_c3=512, f_c2=256, n_class=2):
        super().__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Project Transformer Output
        self.proj_t = conv_block(t_in, 512)
        
        # Fusion 1: Transformer Out (1/16) -> Upsample -> Fuse with C3 (1/8)
        # skipping explicit fusion with C4 because t_out IS the processed C4
        self.up_c3 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.conv_c3 = conv_block(512 + f_c3, 256) 

        # Fusion 2: C3 -> Upsample -> Fuse with C2 (1/4)
        self.up_c2 = nn.Upsample(scale_factor=2, mode='nearest') 
        self.conv_c2 = conv_block(256 + f_c2, 128) 

        # Final Upsample (1/4 -> Full)
        self.up_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.output = nn.Conv2d(128, n_class, kernel_size=1)

    def forward(self, t_out, f_c4, f_c3, f_c2):
        # Arguments:
        # t_out: Transformer output (1024, 16, 16)
        # f_c4:  ResNet C4 diff (1024, 16, 16) -- We won't use this, t_out replaces it
        # f_c3:  ResNet C3 diff (512, 32, 32)
        # f_c2:  ResNet C2 diff (256, 64, 64)
        
        # Level 1: Transformer Out -> Upsample
        f = self.proj_t(t_out)
        f = self.up_c3(f)
        
        # Level 2: Concatenate with C3
        f = torch.cat([f, f_c3], dim=1)
        f = self.conv_c3(f)

        # Level 3: Upsample & Concatenate with C2
        f = self.up_c2(f)
        f = torch.cat([f, f_c2], dim=1)
        f = self.conv_c2(f)

        # Final Prediction
        f = self.up_final(f)
        return self.output(f)

# --- Step 4: The Main Model ---
class SimplifiedBIT(nn.Module):
    def __init__(self, num_classes=2, transformer_dim=1024):
        super().__init__()
        
        self.backbone = FeatureExtractor() 
        
        # Project C4 features (1024 ch) to Transformer dim
        self.c4_proj = nn.Conv2d(1024, transformer_dim, kernel_size=1) 
        
        self.transformer = BIT_Transformer(in_channels=transformer_dim, 
                                           transformer_dim=transformer_dim)
        
        # Initialize Decoder
        self.decoder = UNetDecoder(t_in=transformer_dim,
                                   f_c4=1024,
                                   f_c3=512,
                                   f_c2=256,
                                   n_class=num_classes)

    def forward(self, image_A, image_B):
        # 1. Extract multi-level features
        f_c2_A, f_c3_A, f_c4_A = self.backbone(image_A)
        f_c2_B, f_c3_B, f_c4_B = self.backbone(image_B)

        # 2. Transformer Context (using deep features)
        t_in_A = self.c4_proj(f_c4_A)
        t_in_B = self.c4_proj(f_c4_B)
        t_out = self.transformer(t_in_A, t_in_B)
        
        # 3. Compute Absolute Differences for Skip Connections
        # This prevents "hallucinating" static buildings
        diff_c4 = torch.abs(f_c4_A - f_c4_B)
        diff_c3 = torch.abs(f_c3_A - f_c3_B)
        diff_c2 = torch.abs(f_c2_A - f_c2_B)
        
        # 4. Decode
        change_map = self.decoder(t_out, diff_c4, diff_c3, diff_c2)
        
        return change_map