from einops import rearrange, reduce
from timm.models.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_, create_attn
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE
from model.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D, MSPatchEmb

from model import MODEL

import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn

inplace = True


class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False,
                 conv_branch=False, downsample_skip=False, shuffle=False, conv_local=True):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = has_skip
        self.attn_s = attn_s
        self.conv_branch = conv_branch
        self.shuffle = shuffle
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(in_channels=dim_in, 
                                out_channels=dim_in*2, 
                                kernel_size=1,
                                stride=1,
                                padding="same",
                                bias=qkv_bias)
            self.v = nn.Conv2d(in_channels=dim_in, 
                               out_channels=dim_mid, 
                               kernel_size=1,
                               stride=1,
                               padding="same",
                               bias=qkv_bias)
                               
            #self.attn_drop = nn.Dropout(attn_drop)
            self.drop = attn_drop
            
            if self.conv_branch:
                dim = dim_in if attn_pre else dim_mid
                self.attn_weight = nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv3 = nn.Conv2d(in_channels=dim, 
                                       out_channels=dim,
                                       kernel_size=3,
                                       stride=1,
                                       padding="same",
                                       groups=dim)
                self.conv3_weight = nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv5 = nn.Conv2d(in_channels=dim, 
                                       out_channels=dim, 
                                       kernel_size=5,
                                       stride=1,
                                       padding="same",
                                       groups=dim)
                self.conv5_weight = nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv7 = nn.Conv2d(in_channels=dim, 
                                       out_channels=dim, 
                                       kernel_size=7,
                                       stride=1,
                                       padding="same",
                                       groups=dim)
                self.conv7_weight = nn.Parameter(torch.rand((1, dim, 1, 1)))
                
        else:
            if v_proj:
                self.v = nn.Conv2d(in_channels=dim_in, 
                                   out_channels=dim_mid, 
                                   kernel_size=1,
                                   stride=1,
                                   padding="same",
                                   bias=qkv_bias)
            else:
                self.v = nn.Identity()        
         
        if conv_local or stride > 1:  
            self.conv_local = ConvNormAct(dim_mid, 
                                          dim_mid, 
                                          kernel_size=dw_ks, 
                                          stride=stride, 
                                          dilation=dilation, 
                                          groups=dim_mid, 
                                          norm_layer='none', 
                                          act_layer='none', 
                                          inplace=inplace)
            self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()
        else:
            self.conv_local = nn.Identity()
            self.se = nn.Identity()
            
        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(in_channels=dim_mid, 
                              out_channels=dim_out, 
                              kernel_size=1,
                              stride=1,
                              padding="same",
                              bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.act = get_act(act_layer)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.shuffle:
            x = x.reshape(B, C, 2, H//2, 2, W//2).permute(0,1,3,2,5,4).reshape(B, C, H, W)
        
        # Case 1: Normal layer
        if self.attn_s:
            # Convert x to window-based x
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            
            # Window-based x shape
            b, c, h, w = x.shape
            
            # Calculate Query (Q) and Key (K)
            qk = self.qk(x) # b, 2c, h, w
            qk = qk.reshape(b, 2, self.num_head, c//self.num_head, h*w).permute(1, 0, 2, 4, 3).contiguous() # 2, b, nh, h*w, c//nh
            q, k = qk[0], qk[1] # b, nh, h*w, c//nh
            
            # Case 1: Fuse tokens BEFORE the Value (V) projection (dimension expansion)
            if self.attn_pre:
                # Self-attention
                x_spa = F.scaled_dot_product_attention(query = q,
                                                       key = k,
                                                       value = x.reshape(b, self.num_head, c//self.num_head, h*w).transpose(-1, -2).contiguous(),
                                                       dropout_p = self.drop) # b, nh, h*w, c//nh
                x_spa = x_spa.transpose(-1,-2).reshape(b, c, h, w) # b, c, h, w
                
                if self.conv_branch:
                    # Depth-wise convolutions
                    x_conv3 = self.conv3(x).contiguous() # b, c, h, w
                    x_conv5 = self.conv5(x).contiguous() # b, c, h, w
                    x_conv7 = self.conv7(x).contiguous() # b, c, h, w
                    
                    # Fuse the outputs, with shortcut
                    x = x_spa * self.attn_weight + \
                        x_conv3 * self.conv3_weight + \
                        x_conv5 * self.conv5_weight + \
                        x_conv7 * self.conv7_weight + \
                        x # b, c, h, w
                else:
                    x = x_spa + x if self.has_skip else x_spa # b, c, h, w
                
                # Calculate Value (V)
                x = self.v(x) # b, c_mid, h, w
                
                # Calculate local convolution
                if self.conv_local is not nn.Identity():
                    x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x)) # b, c_mid, h, w
            
            # Case 2: Fuse tokens AFTER the Value (V) projection (dimension expansion)    
            else:
                # Calculate Value (V)
                x = self.v(x) # b, c_mid, h, w
                
                # new x shape
                c_mid = x.shape[1]
                
                # Self-attention
                x_spa = F.scaled_dot_product_attention(query = q,
                                                       key = k,
                                                       value = x.reshape(b, self.num_head, c_mid//self.num_head, h*w).transpose(-1, -2).contiguous(),
                                                       dropout_p = self.drop) # b, nh, h*w, c_mid//nh
                x_spa = x_spa.transpose(-1,-2).reshape(b, c_mid, h, w) # b, c_mid, h, w
                
                if self.conv_branch:
                    # Depth-wise convolutions
                    x_conv3 = self.conv3(x).contiguous() # b, c_mid, h, w
                    x_conv5 = self.conv5(x).contiguous() # b, c_mid, h, w
                    x_conv7 = self.conv7(x).contiguous() # b, c_mid, h, w
                    
                    # Fuse the outputs, with shortcut
                    x = x_spa * self.attn_weight + \
                        x_conv3 * self.conv3_weight + \
                        x_conv5 * self.conv5_weight + \
                        x_conv7 * self.conv7_weight + \
                        x # b, c_mid, h, w
                else:
                    x = x + x_spa if self.has_skip else x_spa # b, c_mid, h, w

            # Convert x to original x
            x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
                
        # Case 2: Downsampling layer
        else:
            x = self.v(x)
            # Calculate local convolution
            if self.conv_local is not nn.Identity():
                x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x)) # b, c_mid, h, w
        
        # Reduce dimension
        x = self.proj_drop(x)
        x = self.proj(x)
        x = shortcut + self.drop_path(x) if self.has_skip else self.drop_path(x)
        self.act(x) # post_activation
        
        # Shuffle back
        if self.shuffle:
            x = x.reshape(B, C, H//2, 2, W//2, 2).permute(0,1,3,2,5,4).reshape(B, C, H, W)
            
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
                    has_skip = False
                    attn_s = False
                    exp_ratio = exp_ratios[i] * 2
                    shuffle_type = False
                    conv_local_type = True
                else:
                    stride = 1
                    has_skip = True
                    attn_s = attn_ss[i]
                    exp_ratio = exp_ratios[i]
                    shuffle_type = True if ((i<len(depths)-1) and (j%2==0 and shuffle)) else False
                    conv_local_type = conv_local
                        
                layers.append(iRMB(emb_dim_pre, embed_dims[i], norm_in=True, has_skip=has_skip, exp_ratio=exp_ratio,
                                   norm_layer=norm_layers[i], act_layer=act_layers[i], v_proj=True, dw_ks=dw_kss[i],
                                   stride=stride, dilation=1, se_ratio=se_ratios[i],
                                   dim_head=dim_heads[i], window_size=window_sizes[i], attn_s=attn_s,
                                   qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, drop_path=dpr[j], v_group=v_group,
                                   attn_pre=attn_pre, conv_branch=conv_branchs[i], downsample_skip=downsample_skip, 
                                   shuffle=shuffle_type, conv_local=conv_local_type))
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
        for blk in self.stage0:
            if self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage1:
            if self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage2:
            if self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage3:
            if self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage4:
            if self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
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
def EMO_6M_4BranchInStage4(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, True], shuffle=False,
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_4BranchInStage34(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, True, True], shuffle=False,
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_4BranchInStage234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, True, True, True], shuffle=False,
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_4BranchInStage1234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, True, True, True], shuffle=False,
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_4BranchInStage1(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, False, False, False], shuffle=False,
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_4BranchInStage12(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, True, False, False], shuffle=False,
                **kwargs)
    return model

@MODEL.register_module
def EMO_6M_WindowShuffle(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=True, 
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_DeeperNarrower(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 12, 4], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 3., 3.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, 
                **kwargs)
    return model
    
@MODEL.register_module
def EMO_6M_DownsampleSkip(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=True, conv_branchs=[False, False, False, False], shuffle=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention_DeeperNarrower(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[2, 2, 10, 2], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage4(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage34(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, True, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, True, True, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage1234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, True, True, True], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage1(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, False, False, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    

@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage12(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, True, False, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_4BranchInStage123(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 6], stem_dim=24, embed_dims=[32, 64, 128, 256], exp_ratios=[3., 4., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[True, True, True, False], shuffle=False, conv_local=False, 
                **kwargs)
    return model
    
    
@MODEL.register_module
def EMO_6M_AllSelfAttention_7x7Kernel_test(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 10, 3], stem_dim=24, embed_dims=[48, 64, 128, 256], exp_ratios=[3., 4., 4., 4.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[7, 7, 7, 7], dim_heads=[16, 16, 32, 32], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, conv_local=True, 
                **kwargs) 
    return model
    
    
@MODEL.register_module
def EMO_6M_SelfAttentionInStage234_4BranchInStage234_PostActivation_PostAttn(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[14, 14, 7, 7], attn_ss=[False, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, True, True, True], shuffle=False, conv_local=False, # True, True, True, True
                **kwargs) 
    return model
    

@MODEL.register_module
def EMO_6M_SelfAttentionInStage234_4BranchInStage34_PostActivation_PostAttn(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[14, 14, 7, 7], attn_ss=[False, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, True, True], shuffle=False, conv_local=False, # True, True, True, True
                **kwargs) 
    return model
    
    
@MODEL.register_module
def EMO_6M_SelfAttentionInStage234_4BranchInStage4_PostActivation_PostAttn(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[14, 14, 7, 7], attn_ss=[False, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, True], shuffle=False, conv_local=False, # True, True, True, True
                **kwargs) 
    return model

    
@MODEL.register_module
def EMO_6M_SelfAttentionInStage234_PostActivation_PostAttn(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'], act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[14, 14, 7, 7], attn_ss=[False, True, True, True],
                qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=False, pre_dim=0,
                downsample_skip=False, conv_branchs=[False, False, False, False], shuffle=False, conv_local=False, # True, True, True, True
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
	# 	blk.attn_pre = False
	# for blk in fn1.stage1:
	# 	blk.attn_pre = False
	# for blk in fn1.stage2:
	# 	blk.attn_pre = False
	# for blk in fn1.stage3:
	# 	blk.attn_pre = False
	# for blk in fn1.stage4:
	# 	blk.attn_pre = False
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