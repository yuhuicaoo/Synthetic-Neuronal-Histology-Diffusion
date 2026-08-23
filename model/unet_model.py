import torch
import torch.nn as nn
from config import DiffusionConfig
import math
from model.attention import MultiHeadAttention, SelfAttention2D

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Creates an embedding vector from a timestep t through a series of sin() and cos() functions with different progressing frequencies
        """
        device = time.device

        # split embedding into halves, half for sin() and half for cos() 
        half_dim = self.dim // 2
        # create frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # each timestep gets multiplied by all frequencies
        embeddings = time[:, None] * embeddings[None, :]                        # (Batch_size, half_dim)
        # apply sin() and cos()
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)    # (Batch_size, dim)
        return embeddings

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4,stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4,stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.norm1 = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.norm2 = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.silu = nn.SiLU()

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, t):
        # first convolution
        h = self.silu(self.norm1(self.conv1(x)))

        # time embeddings injection
        time_emb = self.silu(self.time_mlp(t))
        h = h+ time_emb[(...,) + (None,) * 2]       # broadcast time_emb:(B, C) -> (B, C, H, W)
        # second convolution
        h = self.silu(self.norm2(self.conv2(h)))

        # residual
        return h + self.residual_proj(x)

class Bottleneck(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()

        self.block1 = ResNetBlock(channels, channels, time_emb_dim)
        self.attn = SelfAttention2D(channels)
        self.block2 = ResNetBlock(channels, channels, time_emb_dim)
    
    def forward(self, x , cond):
        x = self.block1(x, cond)
        x = self.attn(x)
        x = self.block2(x, cond)
        return x

class ConditionedUnet(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        in_channels = config.in_channels
        output_dim = config.in_channels
        time_emb_dim = config.time_emb_dim
        
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        self.num_treatments = config.num_treatments
        self.num_regions = config.num_regions

        # + 1 to num_treatments & num_regions to represent null token for CFG
        self.treatment_emb = nn.Embedding(config.num_treatments + 1, config.time_emb_dim)
        self.region_emb = nn.Embedding(config.num_regions + 1, config.time_emb_dim)

        # convert timestep to a feature vector of size time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        # concatenate treatment and region conditioning to timestep
        self.cond_proj = nn.Sequential(
            nn.Linear(time_emb_dim * 3, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=down_channels[0], kernel_size=3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList([
            ResNetBlock(
                in_channels = down_channels[i], 
                out_channels = down_channels[i+1], 
                time_embedding_dim = time_emb_dim
            ) for i in range(len(down_channels) - 1)
        ])
        self.downsamples = nn.ModuleList([Downsample(down_channels[i + 1]) for i in range(len(down_channels) - 1)])

        # Bottleneck (ResnetBlock -> Attention -> ResnetBlock)
        self.bottleneck = Bottleneck(down_channels[-1], time_emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList([
            ResNetBlock(
                in_channels = up_channels[i] * 2,       # *2 for skip connection concatenation 
                out_channels = up_channels[i + 1], 
                time_embedding_dim = time_emb_dim, 
            ) for i in range(len(up_channels) - 1)
        ])
        self.upsamples = nn.ModuleList([Upsample(up_channels[i]) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], output_dim, kernel_size = 1)

    def forward(self, x, timestep, treatment_label, region_label):
        t = self.time_mlp(timestep)                                                 # (B, time_emb_size)
        treatment_emb = self.treatment_emb(treatment_label)                         # (B, time_emb_size)
        region_emb = self.region_emb(region_label)                                  # (B, time_emb_size)
        cond = self.cond_proj(torch.cat((t, treatment_emb, region_emb), dim=1))     # (B, time_emb_size)
        
        x = self.conv0(x)

        residual_inputs = []
        # Encoder
        for block, downsample in zip(self.down_blocks, self.downsamples):
            x = block(x, cond)
            residual_inputs.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.bottleneck(x, cond)

        # Decoder
        for block, upsample in zip(self.up_blocks, self.upsamples):
            x = upsample(x)
            residual_x = residual_inputs.pop()
            # concatenate skip connections to inputs
            x = torch.cat((x, residual_x), dim=1)
            x = block(x, cond)
            
        return self.output(x)
    
class ConditionedUnet_Image(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        in_channels = config.in_channels
        output_dim = config.in_channels
        time_emb_dim = config.time_emb_dim
        
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        self.conv0 = nn.Conv2d(in_channels=in_channels * 2, out_channels=down_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([ResNetBlock(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        self.downsamples = nn.ModuleList([Downsample(down_channels[i+1]) for i in range(len(down_channels) - 1)])

        self.bottleneck_block1 = ResNetBlock(down_channels[-1], down_channels[-1], time_emb_dim)
        self.bottleneck_attn = SelfAttention2D(down_channels[-1], num_heads=8)
        self.bottleneck_block2 = ResNetBlock(down_channels[-1], down_channels[-1], time_emb_dim)

        self.up_blocks = nn.ModuleList([ResNetBlock(up_channels[i] * 2, up_channels[i+1], time_emb_dim) for i in range(len(up_channels) - 1)])
        self.upsamples = nn.ModuleList([Upsample(up_channels[i]) for i in range(len(up_channels) -1)])

        self.output = nn.Conv2d(up_channels[-1], output_dim, kernel_size = 1)

    def forward(self, x, timestep, mask):
        t = self.time_mlp(timestep)

        x = self.conv0(torch.cat([x, mask], dim=1))

        residual_inputs = []
        for down_block, downsample in list(zip(self.down_blocks, self.downsamples)):
            x = down_block(x, t)
            residual_inputs.append(x)
            x = downsample(x)
        
        x = self.bottleneck_block1(x, t)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_block2(x, t)

        for up_block, upsample in list(zip(self.up_blocks, self.upsamples)):
            x = upsample(x)
            residual_x = residual_inputs.pop()
            # concatenate skip connections to inputs
            x = torch.cat((x, residual_x), dim=1)
            x = up_block(x, t)
        return self.output(x)