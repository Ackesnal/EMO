from einops import rearrange, reduce
from timm.models.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_, create_attn
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE
from model.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D, MSPatchEmb

from functools import partial
from model import MODEL
import math
import torch.utils.checkpoint as ckpt
import torch.nn.functional as F
import torch.nn as nn

inplace = True

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        # 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False,
                 conv_branch=False, downsample_skip=False, shuffle=False):
        super().__init__()
        dim_mid = int(dim_in * exp_ratio)
        self.attn_s = attn_s
        self.conv_branch = conv_branch
        self.downsample = True if stride > 1 else False
        self.shuffle = shuffle
        self.norm = nn.BatchNorm2d(dim_in)
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.attn_pre = attn_pre
            self.qkv = nn.Conv2d(in_channels=dim_in, 
                                out_channels=dim_in*3, 
                                kernel_size=1,
                                stride=1,
                                padding="same",
                                bias=qkv_bias)
                               
            self.drop = attn_drop
            
            if self.conv_branch:
                self.attn_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv3 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in)
                self.conv3_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv5 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in, 
                                       kernel_size=5,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in)
                self.conv5_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv7 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in, 
                                       kernel_size=7,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in)
                self.conv7_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
            
            # FFN with convolution
            self.ffn_in = nn.Conv2d(in_channels=dim_in, 
                                    out_channels=dim_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding="same",
                                    groups=1)
            self.conv_local = nn.Conv2d(in_channels=dim_in,
                                        out_channels=dim_in,
                                        kernel_size=dw_ks,
                                        stride=stride,
                                        padding="same",
                                        dilation=dilation,
                                        groups=dim_in)
            self.ffn_out = nn.Conv2d(in_channels=dim_in,
                                     out_channels=dim_out,
                                     kernel_size=1,
                                     stride=1,
                                     padding="same",
                                     groups=1)
            self.ffn_act = nn.SiLU()
            # Final drop path
            self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
                
        else:
            # FFN with convolution
            self.ffn_in = nn.Conv2d(in_channels=dim_in, 
                                    out_channels=dim_in, 
                                    kernel_size=1,
                                    stride=1,
                                    padding="same",
                                    groups=1,
                                    bias=qkv_bias)
            padding=math.ceil((dw_ks-stride)/2)
            self.conv_local = nn.Conv2d(in_channels=dim_in,
                                        out_channels=dim_out,
                                        kernel_size=dw_ks,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=padding,
                                        groups=dim_in,
                                        bias=qkv_bias)
            self.ffn_out = nn.Conv2d(in_channels=dim_out,
                                     out_channels=dim_out,
                                     kernel_size=1,
                                     stride=1,
                                     padding="same",
                                     groups=1,
                                     bias=qkv_bias)
            self.ffn_act = nn.SiLU()
            # Final drop path
            self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        if self.training:
            B, C, H, W = x.shape
            
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                # Pre-layer normalization
                x = self.norm(x)
                
                # Convert x to window-based x
                window_size_W, window_size_H = self.window_size, self.window_size
                pad_l, pad_t = 0, 0
                pad_r = (window_size_W - W % window_size_W) % window_size_W
                pad_b = (window_size_H - H % window_size_H) % window_size_H
                x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
                n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
                x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2)#.contiguous()
                
                # Window-based x shape
                b, c, h, w = x.shape
                
                # Calculate Query (Q) and Key (K)
                qkv = self.qkv(x) # b, 3c, h, w
                qkv = qkv.reshape(b, 3, self.num_head, c//self.num_head, h*w).permute(1, 0, 2, 4, 3).contiguous() # 3, b, nh, h*w, c//nh
                q, k, v = qkv[0], qkv[1], qkv[2] # b, nh, h*w, c//nh
                
                # Add shortcut 1
                v = v + x.reshape(b, self.num_head, c//self.num_head, h*w).transpose(-1,-2)  # b, nh, h*w, c//nh
                
                # Self-attention
                x_spa = F.scaled_dot_product_attention(query = q,
                                                       key = k,
                                                       value = v,
                                                       dropout_p = self.drop) # b, nh, h*w, c//nh
                                                                                                          
                x_spa = x_spa.transpose(-1,-2).reshape(b, c, h, w) # b, c, h, w
                
                if self.conv_branch:
                    # Depth-wise convolutions
                    x_conv3 = self.conv3(x).contiguous() # b, c_mid, h, w
                    x_conv5 = self.conv5(x).contiguous() # b, c_mid, h, w
                    x_conv7 = self.conv7(x).contiguous() # b, c_mid, h, w
                        
                    # Fuse the outputs, with shortcut
                    x_spa = x_spa * self.attn_weight + \
                            x_conv3 * self.conv3_weight + \
                            x_conv5 * self.conv5_weight + \
                            x_conv7 * self.conv7_weight # b, c, h, w
                
                    
                # Add shortcut 2
                x = x_spa + v.transpose(-1,-2).reshape(b, c, h, w)
    
                # Convert x to original x
                x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2)#.contiguous()
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W] # B, C, H, W
                
                
                # FFN
                shortcut = x
                x = self.ffn_in(x)
                x = self.conv_local(x)
                x = self.ffn_act(x)
                x = self.ffn_out(x)
                
                # Drop path and shortcut
                x = self.drop_path(x) + shortcut
                
                return x
                    
            # Case 2: Downsampling layer
            else:
                if self.downsample:
                    # FFN
                    x = self.norm(x)
                    x = self.ffn_in(x)
                    x = self.conv_local(x)
                    x = self.ffn_act(x)
                    x = self.ffn_out(x)
                    # Drop path
                    x = self.drop_path(x)
                    return x
                else:
                    shortcut = x
                    # FFN
                    x = self.norm(x)
                    x = self.ffn_in(x)
                    x = self.conv_local(x)
                    x = self.ffn_act(x)
                    x = self.ffn_out(x)
                    # Drop path
                    x = self.drop_path(x) + shortcut
                    return x
        
        else:
            B, C, H, W = x.shape
            
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                # Convert x to window-based x
                window_size_W, window_size_H = self.window_size, self.window_size
                pad_l, pad_t = 0, 0
                pad_r = (window_size_W - W % window_size_W) % window_size_W
                pad_b = (window_size_H - H % window_size_H) % window_size_H
                x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
                n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
                x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2)#.contiguous()
                
                # Window-based x shape
                b, c, h, w = x.shape
                
                # Calculate Query (Q) and Key (K)
                qkv = self.qkv(x) # b, 3c, h, w
                qkv = qkv.reshape(b, 3, self.num_head, c//self.num_head, h*w).permute(1, 0, 2, 4, 3).contiguous() # 3, b, nh, h*w, c//nh
                q, k, v = qkv[0], qkv[1], qkv[2] # b, nh, h*w, c//nh
                
                # Self-attention
                x_spa = F.scaled_dot_product_attention(query = q,
                                                       key = k,
                                                       value = v,
                                                       dropout_p = self.drop) # b, nh, h*w, c//nh
                                                                                                          
                x_spa = x_spa.transpose(-1,-2).reshape(b, c, h, w) # b, c, h, w
                
                # Convert x to original x
                x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2)#.contiguous()
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :, :H, :W] # B, C, H, W
                
                # FFN
                shortcut = x
                x = self.ffn_in(x)
                x = self.conv_local(x)
                x = self.ffn_act(x)
                x = self.ffn_out(x)
                
                # Drop path and shortcut
                x = self.drop_path(x) + shortcut
                
                return x
                    
            # Case 2: Downsampling layer
            else:
                if self.downsample:
                    # FFN
                    x = self.ffn_in(x)
                    x = self.conv_local(x)
                    x = self.ffn_act(x)
                    x = self.ffn_out(x)
                    # Drop path
                    x = self.drop_path(x)
                    return x
                else:
                    shortcut = x
                    # FFN
                    x = self.ffn_in(x)
                    x = self.conv_local(x)
                    x = self.ffn_act(x)
                    x = self.ffn_out(x)
                    # Drop path
                    x = self.drop_path(x) + shortcut
                    return x


class EMO(nn.Module):
    def __init__(self, dim_in=3, num_classes=1000, img_size=224, depths=[1, 2, 4, 2], stem_dim=16,
                 embed_dims=[64, 128, 256, 512], exp_ratios=[4., 4., 4., 4.], 
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], act_layers=['relu', 'relu', 'relu', 'relu'],
                 dw_kss=[3, 3, 5, 5], se_ratios=[0.0, 0.0, 0.0, 0.0], dim_heads=[32, 32, 32, 32],
                 window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True], qkv_bias=True,
                 attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False, pre_dim=0,
                 conv_branchs=[False, False, False, False], downsample_skip=False, shuffle=False,
                 conv_local=True):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([MSPatchEmb(dim_in, stem_dim, kernel_size=dw_kss[0], 
                                                c_group=1, stride=2, dilations=[1, 2, 3],
                                                norm_layer=norm_layers[0], act_layer='none')])
        emb_dim_pre = stem_dim
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride = 2
                    attn_s = False
                    exp_ratio = exp_ratios[i] * 2
                    shuffle_type = False
                else:
                    stride = 1
                    attn_s = attn_ss[i]
                    exp_ratio = exp_ratios[i]
                    shuffle_type = True if ((i<len(depths)-1) and (j%2==0 and shuffle)) else False
                        
                layers.append(iRMB(emb_dim_pre, embed_dims[i], norm_in=True, exp_ratio=exp_ratio,
                                   norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
                                   stride=stride, dilation=1, se_ratio=se_ratios[i],
                                   dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
                                   qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
                                   attn_pre=attn_pre, conv_branch=conv_branchs[i], downsample_skip=downsample_skip, 
                                   shuffle=shuffle_type))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
        
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        if pre_dim > 0:
            self.pre_head = nn.Sequential(nn.Linear(embed_dims[-1], pre_dim), get_act(act_layers[-1])(inplace=inplace))
            self.pre_dim = pre_dim
        else:
            self.pre_head = nn.Identity()
            self.pre_dim = embed_dims[-1]
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, 
                            nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, 
                            nn.InstanceNorm2d, nn.InstanceNorm3d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'alpha', 'gamma', 'beta'}

    @torch.jit.ignore
    def no_ft_keywords(self):
        # return {'head.weight', 'head.bias'}
        return {}

    @torch.jit.ignore
    def ft_head_keywords(self): 
        return {'head.weight', 'head.bias'}, self.num_classes

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.pre_dim, num_classes) if num_classes > 0 else nn.Identity()

    def check_bn(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.modules.batchnorm._NormBase):
                m.running_mean = torch.nan_to_num(m.running_mean, nan=0, posinf=1, neginf=-1)
                m.running_var = torch.nan_to_num(m.running_var, nan=0, posinf=1, neginf=-1)

    def forward_features(self, x):
        if self.training:
            for blk in self.stage0:
                x = ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage1:
                x = ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage2:
                x = ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage3:
                x = ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage4:
                x = ckpt.checkpoint(blk, x.requires_grad_(True))
        else:
            for blk in self.stage0:
                x = blk(x)
            for blk in self.stage1:
                x = blk(x)
            for blk in self.stage2:
                x = blk(x)
            for blk in self.stage3:
                x = blk(x)
            for blk in self.stage4:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
        x = self.pre_head(x)
        x = self.head(x)
        return {'out': x, 'out_kd': x}


        

@MODEL.register_module
def EMO_1M(pretrained=False, **kwargs):
    model = EMO(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[2, 2, 8, 3], stem_dim=24, embed_dims=[32, 48, 80, 168], exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 21], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036, v_group=False, attn_pre=True, pre_dim=0,
        **kwargs)
    return model

@MODEL.register_module
def EMO_2M(pretrained=False, **kwargs):
    model = EMO(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[32, 48, 120, 200], exp_ratios=[2., 2.5, 3.0, 3.5],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 20, 20], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
        **kwargs)
    return model

@MODEL.register_module
def EMO_5M(pretrained=False, **kwargs):
    model = EMO(
        # dim_in=3, num_classes=1000, img_size=224,
        depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 288], exp_ratios=[2., 3., 4., 4.],
        norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
        dw_kss=[3, 3, 5, 5], dim_heads=[24, 24, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
        **kwargs)
    return model

@MODEL.register_module
def EMO_6M(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], exp_ratios=[3., 3., 3., 3.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage4(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], exp_ratios=[3., 3., 3., 3.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage34(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], exp_ratios=[3., 3., 3., 3.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, True, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
 


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    import copy
    import time
    
    
    def get_timepc():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
    
    
    def get_net_params(net):
        num_params = 0
        for param in net.parameters():
            if param.requires_grad:
                num_params += param.numel()
        return num_params / 1e6


    bs = 2
    reso = 224
    # reso = 256
    x = torch.randn(bs, 3, reso, reso).cuda()
    fn = EMO_1M().cuda()
    # fn = EMO_2M().cuda()
    # fn = EMO_5M().cuda()
    # fn = EMO_6M().cuda()

    fn.eval()
    y = fn(x)
    print(y['out'])
    
    # fn1 = copy.deepcopy(fn)
    # for blk in fn1.stage0:
    #     blk.attn_pre = False
    # for blk in fn1.stage1:
    #     blk.attn_pre = False
    # for blk in fn1.stage2:
    #     blk.attn_pre = False
    # for blk in fn1.stage3:
    #     blk.attn_pre = False
    # for blk in fn1.stage4:
    #     blk.attn_pre = False
    # y1 = fn1(x)
    # print(y1['out'])
    
    flops = FlopCountAnalysis(fn, torch.randn(1, 3, 224, 224).cuda())
    print(flop_count_table(flops, max_depth=3))
    flops = FlopCountAnalysis(fn, x).total() / bs / 1e9
    params = parameter_count(fn)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = fn(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = fn(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))
# print(flop_count_table(FlopCountAnalysis(fn, x), max_depth=3))