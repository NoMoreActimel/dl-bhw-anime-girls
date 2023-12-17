import math
import torch
from torch import nn

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.layers = nn.Sequential(
            nn.Linear(self.n_channels // 4, self.n_channels),
            SwishActivation(),
            nn.Linear(self.n_channels, self.n_channels)
        )

    def forward(self, t):
        embed_dim = self.n_channels // 8
        emb = math.log(10_000) / (embed_dim - 1)
        emb = torch.exp(torch.arange(embed_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.layers(emb)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups=32, dropout=0.1):
        super().__init__()

        self.layers1 = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            SwishActivation(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )
        self.layers2 = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            SwishActivation(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        )
        self.time_layers = nn.Sequential(
            SwishActivation(),
            nn.Linear(time_channels, out_channels)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.layers1(x)
        h = h + self.time_layers(t)[:, :, None, None]
        h = self.layers2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, n_channels, n_heads=1, d_k=None, n_groups=32):
        super().__init__()

        d_k = n_channels if d_k is None else d_k
        self.d_k = d_k
        
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads

    def forward(self, x, t=None):
        batch_size, n_channels, height, width = x.shape
        
        x = x.view(batch_size, n_channels, -1).transpose(1, 2)
        # batch_size x (height * width) x n_channels

        qkv = self.projection(x)
        qkv = qkv.view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = res + x

        res = res.transpose(1, 2)
        # batch_size x n_channels x (height * width)
        res = res.view(batch_size, n_channels, height, width)

        return res


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, image_channels = 3, n_channels = 64,
                 ch_mults = (1, 2, 2, 4),
                 is_attn = (False, False, True, True),
                 n_blocks = 2):
        super().__init__()

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels

            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))

            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = SwishActivation()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)
        h = [x]
        
        for m in self.down:
            x = m(x, t)
            h.append(x)

        x = self.middle(x, t)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        return self.final(self.act(self.norm(x)))
