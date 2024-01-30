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
import copy
from flash_attn.flash_attn_triton import _flash_attn_forward

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
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_layer='bn_2d', act_layer='relu', 
                 dw_ks=3, stride=1, dilation=1, dim_head=64, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop_path=0.,
                 conv_branch=False, alpha=0.1, beta=0.2, theta=0.5, H=0, W=0, window_input=False):
        super().__init__()
        self.attn_s = attn_s
        self.conv_branch = conv_branch
        self.window_size = window_size
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.scale = dim_head ** (-0.5)
            self.dim_in = dim_in
            self.num_head = dim_in // dim_head
            self.qkv = nn.Linear(in_features=dim_in, out_features=dim_in*3)
            self.attn_mask = 0
            self.conv_mask = 0
            self.attn_weight = 1
            self.alpha = alpha
            self.beta = beta
            self.theta = theta
            self.drop = attn_drop
            
            if self.conv_branch:
                self.attn_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv3 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in,
                                       bias=qkv_bias)
                self.conv3_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv5 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in, 
                                       kernel_size=5,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in,
                                       bias=qkv_bias)
                self.conv5_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
                
                self.conv7 = nn.Conv2d(in_channels=dim_in, 
                                       out_channels=dim_in, 
                                       kernel_size=7,
                                       stride=1,
                                       padding="same",
                                       groups=dim_in,
                                       bias=qkv_bias)
                self.conv7_weight = 0.25 # nn.Parameter(torch.rand((1, dim, 1, 1)))
            
            # FFN
            self.ffn_in = nn.Linear(in_features=dim_in, out_features=dim_in, bias=qkv_bias)
            self.ffn_out = nn.Linear(in_features=dim_in, out_features=dim_in, bias=qkv_bias)
            self.ffn_act = nn.SiLU()
            
            # drop paths
            self.drop_path_1 = DropPath(drop_path) if drop_path else nn.Identity()
            self.drop_path_2 = DropPath(drop_path) if drop_path else nn.Identity()
            self.drop_path_3 = DropPath(drop_path) if drop_path else nn.Identity()
            
            # Post layer norm
            self.post_norm = nn.LayerNorm(dim_in)
                
        else:
            self.n1_input = H // self.window_size
            self.n2_input = W // self.window_size
            self.n1_output = H // self.window_size // 2
            self.n2_output = W // self.window_size // 2
            self.window_input = window_input
            # FFN with convolution
            self.ffn_in = nn.Conv2d(in_channels=dim_in,
                                    out_channels=dim_in,
                                    kernel_size=1,
                                    stride=1,
                                    groups=dim_in)
            self.conv_local = nn.Conv2d(in_channels=dim_in,
                                        out_channels=dim_out,
                                        kernel_size=dw_ks,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=math.ceil((dw_ks-stride)/2),
                                        groups=dim_in,
                                        bias=qkv_bias)
            self.ffn_act = nn.SiLU(inplace=True)
            self.ffn_out = nn.Conv2d(in_channels=dim_out,
                                     out_channels=dim_out,
                                     kernel_size=1,
                                     stride=1,
                                     groups=dim_out)
            
            # Post layer norm
            self.post_norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        if self.training:
            B, H, W, C = x.shape # B, H, W, C
            
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                # Convert x to window-based x
                window_size_W, window_size_H = self.window_size, self.window_size
                pad_l, pad_t = 0, 0
                pad_r = (window_size_W - W % window_size_W) % window_size_W
                pad_b = (window_size_H - H % window_size_H) % window_size_H
                x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
                n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
                x = rearrange(x, 'b (h1 n1) (w1 n2) c -> (b n1 n2) h1 w1 c', n1=n1, n2=n2) 
                
                # Window-based x shape
                b, h, w, c = x.shape
                
                # Shortcut 1
                shortcut = x.reshape(b, h*w, self.num_head, c//self.num_head).transpose(1,2) # b, nh, h*w, c//nh
                
                # Calculate Query (Q) and Key (K)
                qkv = self.qkv(x) # b, h, w, 3c
                qkv = qkv.reshape(b, h*w, 3, self.num_head, c//self.num_head).permute(2, 0, 3, 1, 4).contiguous() # 3, b, nh, h*w, c//nh
                q, k, v = qkv[0], qkv[1], qkv[2] # b, nh, h*w, c//nh
                
                # Add shortcut 1 and drop path 1
                v = self.drop_path_1(self.alpha * v) + shortcut # b, nh, h*w, c//nh
                
                # Shortcut 2
                shortcut = v.transpose(1,2).reshape(b, h, w, c) # b, h, w, c
                
                # Self-attention
                x_spa = F.scaled_dot_product_attention(query = q,
                                                       key = k,
                                                       value = v,
                                                       dropout_p = self.drop) # b, nh, h*w, c//nh
                                                                                                          
                x_spa = x_spa.transpose(1,2).reshape(b, h, w, c) # b, h, w, c
                
                # Multiple branchs during training
                if self.conv_branch:
                    x_conv = v.transpose(-1,-2).reshape(b, c, h, w) # b, c, h, w
                    # Depth-wise convolutions
                    x_conv3 = self.conv3(x_conv).permute(0,2,3,1) # b, h, w, c
                    x_conv5 = self.conv5(x_conv).permute(0,2,3,1) # b, h, w, c
                    x_conv7 = self.conv7(x_conv).permute(0,2,3,1) # b, h, w, c
                        
                    # Fuse the outputs
                    x_spa = x_spa * self.attn_weight + \
                            x_conv3 * self.conv3_weight + \
                            x_conv5 * self.conv5_weight + \
                            x_conv7 * self.conv7_weight # b, h, w, c
                
                # Add shortcut 2 and drop path 2
                x = self.drop_path_2(self.beta * x_spa) + shortcut # b, h, w, c
                
                # Convert x to original x
                x = rearrange(x, '(b n1 n2) h1 w1 c -> b (h1 n1) (w1 n2) c', n1=n1, n2=n2)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :H, :W, :] # B, H, W, C
                
                # FFN
                # Shortcut 3
                shortcut = x # B, H, W, C
                x = self.ffn_in(x)
                x = self.ffn_act(x)
                x = self.ffn_out(x) # B, H, W, C
                
                # Add shortcut 3 and drop path 3
                x = self.drop_path_3(self.theta * x) + shortcut
                    
                # Add post normalization
                x = self.post_norm(x)
                
                #if x.get_device()==0 and self.qkv.weight.grad is not None:
                #    print(self.ffn_in.weight.grad.mean(), self.ffn_in.weight.grad.max(), self.ffn_in.weight.grad.min())
                #if x.get_device()==0 and self.qkv.weight.grad is not None:
                #    print(self.qkv.weight.grad.mean(), self.qkv.weight.grad.max(), self.qkv.weight.grad.min())
                #if x.get_device()==0:
                #    print(x.mean(), x.max(), x.min())
                return x
                
            # Case 2: Downsampling layer
            else:
                # FFN
                x = x.permute(0,3,1,2) # B, C, H, W
                x = self.ffn_in(x) # B, H, W, C
                x = self.conv_local(x) # B, C, H, W
                x = self.ffn_out(x) # B, H, W, C
                x = x.permute(0,2,3,1) # B, H, W, C
                    
                # Post-layer normalization
                x = self.post_norm(x)
                return x
                    
        else:
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                B, C, N = x.shape # B, N, C
                
                # Calculate Query (Q) and Key (K)
                qkv = self.qkv(x) # B, 3C, N
                qkv = rearrange(qkv, 'b n (T nh hc) -> T (b nh) n hc', T=3, nh=self.num_head)
                q, k, v = qkv[0], qkv[1]*self.scale, qkv[2] # B*nh, N, C//nh
                
                # Calculate reparameterized self-attention
                attn = torch.bmm(q, k.transpose(-2,-1)).softmax(dim=-1)
                attn.mul_(self.attn_weight).add_(self.attn_mask)
                x_spa = torch.bmm(attn, v) # B, nh, N, C//nh
                
                x = rearrange(x_spa, '(b nh) n hc -> b n (nh hc)', nh=self.num_head)
                
                # FFN
                shortcut = x # B, N, C
                self.ffn_act(x) # B, N, C
                x = x + shortcut # B, N, C
                
                x = self.ffn_out(x) # B, N, C
                
                return x
                
            # Case 2: Downsampling layer
            else:
                # Convert x to original x
                if self.window_input:
                    x = rearrange(x, '(b n1 n2) (h w) c -> b c (h n1) (w n2)', n1=self.n1_input, n2=self.n2_input, h=self.window_size)
                        
                # FFN
                x = self.ffn_in(x) # B, C, H, W
                x = self.conv_local(x) # B, 2C, H/2, W/2                    
                x = self.ffn_out(x) # B, 2C, H/2, W/2
                    
                x = rearrange(x, 'b c (h n1) (w n2) -> (b n1 n2) (h w) c', n1=self.n1_output, n2=self.n2_output) 
                
                return x
        
    def reparam(self):
        if self.attn_s:
            C = self.qkv.weight.shape[1] # [C*3, C]
            
            # Reparam first shortcut
            self.qkv.weight.requires_grad_(False)
            self.qkv.weight[C*2:,:] = self.alpha * self.qkv.weight[C*2:,:] + torch.eye(C).to(self.qkv.weight.device)
            self.qkv.bias.requires_grad_(False)
            self.qkv.bias[C*2:] = self.alpha * self.qkv.bias[C*2:]
            
            # Reparam second shortcut
            self.attn_mask = torch.eye(self.window_size**2).to(self.qkv.weight.device)
            self.attn_mask = self.attn_mask.reshape(1,self.window_size**2,self.window_size**2)
            self.conv_mask = self.conv_mask * self.beta
            self.attn_mask = self.attn_mask + self.conv_mask
            self.attn_weight = self.attn_weight * self.beta
            
            # Reparam third shortcut
            self.ffn_out.weight.requires_grad_(False)
            self.ffn_out.weight = self.ffn_out.weight * self.theta
            self.ffn_out.bias.requires_grad_(False)
            self.ffn_out.bias = self.ffn_out.bias * self.theta
            

class EMO(nn.Module):
    def __init__(self, dim_in=3, num_classes=1000, img_size=224, depths=[1, 2, 4, 2], 
                 stem_dim=16, embed_dims=[48, 96, 192, 384], dim_heads=[32, 32, 32, 32],
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'], 
                 act_layers=['relu', 'relu', 'relu', 'relu'], 
                 dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], qkv_bias=True, 
                 attn_ss=[True, True, True, True], attn_drop=0., drop_path=0., pre_dim=0,
                 conv_branchs=[True, True, True, True], conv_local=True):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([MSPatchEmb(dim_in, stem_dim, kernel_size=dw_kss[0], 
                                                c_group=1, stride=2, dilations=[1, 2, 3],
                                                norm_layer=norm_layers[0], act_layer='none')])
        emb_dim_pre = stem_dim
        H=img_size
        W=img_size
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i + 1])]
            for j in range(depths[i]):
                if j == 0:
                    stride = 2
                    attn_s = False
                    H = H//2
                    W = W//2
                    window_input = True if i > 0 else False
                else:
                    stride = 1
                    attn_s = attn_ss[i]
                    window_input = True
                        
                layers.append(iRMB(emb_dim_pre, embed_dims[i], dim_head=dim_heads[i], 
                                   norm_layer=norm_layers[i], act_layer=act_layers[i], 
                                   dw_ks=dw_kss[i], stride=stride, dilation=1, 
                                   window_size=window_sizes[i], attn_s=attn_s,
                                   qkv_bias=qkv_bias, attn_drop=attn_drop, drop_path=dpr[j], 
                                   conv_branch=conv_branchs[i], H=H, W=W, window_input=window_input))
                emb_dim_pre = embed_dims[i]
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
        
        #self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        if pre_dim > 0:
            self.pre_head = nn.Sequential(nn.Linear(embed_dims[-1], pre_dim), get_act(act_layers[-1])(inplace=inplace))
            self.pre_dim = pre_dim
        else:
            self.pre_head = nn.Identity()
            self.pre_dim = embed_dims[-1]
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
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
                x = blk(x) #ckpt.checkpoint(blk, x.requires_grad_(True))
            x = x.permute(0, 2, 3, 1) # B, H, W, C
            for blk in self.stage1:
                x = blk(x) #ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage2:
                x = blk(x) #ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage3:
                x = blk(x) #ckpt.checkpoint(blk, x.requires_grad_(True))
            for blk in self.stage4:
                x = blk(x) #ckpt.checkpoint(blk, x.requires_grad_(True))
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
        #x = self.norm(x)
        if self.training:
            x = reduce(x, 'b h w c-> b c', 'mean')
        else:
            x = reduce(x, 'b n c-> b c', 'mean')
            
        x = self.pre_head(x)
        x = self.head(x)
        return {'out': x, 'out_kd': x}
        
    def reparam(self):
        for blk in self.stage1:
            blk.reparam()
        for blk in self.stage2:
            blk.reparam()
        for blk in self.stage3:
            blk.reparam()
        for blk in self.stage4:
            blk.reparam()


        

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
def FastAllSelfAttention_8M_1G_SingleBranch(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, False, False], conv_local=False,
                **kwargs)
    return model
    

@MODEL.register_module
def FastAllSelfAttention_8M_1G_4BranchInStage4(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, False, True], conv_local=False,
                **kwargs)
    return model
    
    
@MODEL.register_module
def FastAllSelfAttention_8M_1G_4BranchInStage34(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, True, True], conv_local=False,
                **kwargs)
    return model
    
@MODEL.register_module
def FastAllSelfAttention_8M_1G_4BranchInStage234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, True, True, True], conv_local=False,
                **kwargs)
    return model
    
    
@MODEL.register_module
def FastAllSelfAttention_8M_1G_4BranchInStage1234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 17, 7], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[True, True, True, True], conv_local=False,
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