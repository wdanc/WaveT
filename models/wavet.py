import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import DropPath, create_classifier, create_conv2d, create_pool2d, to_2tuple, to_ntuple
from einops import rearrange

from dwt.transform2d import DWTForward

def _cifar_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 100, 'input_size': (3, 32, 32),
        'interpolation': 'bicubic',
        'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010),

        **kwargs
    }

def _imgnet_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        **kwargs
    }

def _rs_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 45, 'input_size': (3, 224, 224),
        'interpolation': 'bicubic', 'crop_pct':0.875,
        'mean': (0.3680, 0.3810, 0.3436), 'std': (0.1454, 0.1356, 0.1320),

        **kwargs
    }

default_cfgs = {
    'wavet': _cifar_cfg(),
    'wavet_img': _imgnet_cfg(),
    'wavet_rs': _rs_cfg()
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WaveEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, embed_layer, patch_size, stride, pad,
                 in_chans, embed_dim, norm_layer=None, flatten=True):
        super().__init__()
        self.embed = embed_layer
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans*4, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=pad)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        B, C, H, W = x.shape
        #--------------------------------------
        x_l, x_hlist = self.embed(x)
        x_l = x_l.unsqueeze(2)
        x_h = x_hlist[-1]
        x = torch.cat((x_l, x_h), dim=2)
        x = x.reshape(B, -1, x_l.size(3), x_l.size(4))
        # --------------------------------------
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class WStage(nn.Module):
    def __init__(self, embed_layer, patch_size, stride, pad, seq_len, num_heads, embed_dim, prev_dim,
                 depth, qkv_bias, mlp_ratio, drop_rate=0.0, attn_drop=0.0, drop_path_rate=[],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        self.wave_embedding = WaveEmbed(embed_layer=embed_layer, patch_size=patch_size, stride=stride, pad=pad,
                             in_chans=prev_dim, embed_dim=embed_dim, norm_layer=norm_layer, flatten=True)
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop, drop_path=drop_path_rate[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

    def forward(self, x):
        x = self.wave_embedding(x)
        h, w = int(x.size(1)**0.5), int(x.size(1)**0.5)
        x += self.pos_embedding
        x = self.blocks(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x



class WaveT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_levels=3, wave='db1', wave_len=2,
                 patch_sizes=(1,1,1), strides=(1,1,1),
                 pads=(0,0,0), embed_dims=(128, 256, 512), num_heads=(4, 8, 16), depths=(2, 2, 20),
                 num_classes=1000, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.0, norm_layer=None, act_layer=None, global_pool='avg'):
        super().__init__()
        patch_sizes = to_ntuple(num_levels)(patch_sizes)
        strides = to_ntuple(num_levels)(strides)
        pads = to_ntuple(num_levels)(pads)
        embed_dims = to_ntuple(num_levels)(embed_dims)
        num_heads = to_ntuple(num_levels)(num_heads)
        depths = to_ntuple(num_levels)(depths)
        self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        self.num_levels = num_levels
        self.seq_lens = []
        feature_size = img_size
        for i in range(num_levels):
            size = math.floor((feature_size - 1) / 2) + wave_len // 2
            size = math.floor((size - patch_sizes[i] + 2 * pads[i]) / strides[i] + 1)
            self.seq_lens.append(int(size ** 2))
            feature_size = size

        dwt = DWTForward(J=1, wave=wave)

        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = in_chans
        for i in range(num_levels):
            dim = embed_dims[i]
            levels.append(WStage(
                embed_layer=dwt, patch_size=patch_sizes[i], stride=strides[i], pad=pads[i],
                seq_len=self.seq_lens[i], num_heads=num_heads[i], embed_dim=dim, prev_dim=prev_dim,
                depth=depths[i], qkv_bias=qkv_bias, mlp_ratio=mlp_ratio, drop_rate=drop_rate,
                attn_drop=attn_drop_rate, drop_path_rate=dp_rates[i],
                act_layer=act_layer, norm_layer=norm_layer))
            prev_dim = dim
        self.levels = nn.Sequential(*levels)

        self.norm = norm_layer(embed_dims[-1])
        self.global_pool, self.head = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.levels(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        """ x shape (B, C, H, W)
        """
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.head(x)



@register_model
def wavet_tiny_cifar100(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1), embed_dims=(128, 192, 192),
                 num_heads=(2, 3, 3), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_small_cifar100(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1), embed_dims=(192, 384, 384),
                 num_heads=(3, 6, 6), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_large_cifar100(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1), embed_dims=(384, 512, 768),
                 num_heads=(6, 8, 12), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_tiny_cifar10(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1),
                  embed_dims=(128, 192, 192),
                 num_heads=(2, 3, 3), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_small_cifar10(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1),
                  embed_dims=(192, 384, 384),
                 num_heads=(3, 6, 6), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_large_cifar10(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(3, 3, 3), strides=(1, 1, 1), pads=(1, 1, 1), embed_dims=(384, 512, 768),
                 num_heads=(6, 8, 12), depths=(3, 6, 3), mlp_ratio=4., qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet']
    return model

@register_model
def wavet_tiny_nwpu(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(128, 192, 192), num_heads=(2, 3, 3), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_rs']
    return model

@register_model
def wavet_small_nwpu(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(192, 384, 384), num_heads=(3, 6, 6), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_rs']
    return model

@register_model
def wavet_large_nwpu(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(384, 512, 768), num_heads=(6, 8, 12), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_rs']
    return model

@register_model
def wavet_tiny_imgnet(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(128, 192, 192), num_heads=(2, 3, 3), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_img']
    return model

@register_model
def wavet_small_imgnet(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(192, 384, 384), num_heads=(3, 6, 6), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_img']
    return model

@register_model
def wavet_large_imgnet(pretrained=False, **kwargs):
    model = WaveT(num_levels=3, patch_sizes=(4,3,3), strides=(4,1,1), pads=(0,1,1),
                 embed_dims=(384, 512, 768), num_heads=(6, 8, 12), depths=(3, 6, 3), mlp_ratio=4.,
                 qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['wavet_img']
    return model