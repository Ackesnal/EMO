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
            
            # KQV
            self.q_weight = nn.Parameter(torch.rand((dim_in, dim_in)))
            self.q_bias = nn.Parameter(torch.rand((dim_in)))
            self.k_weight = nn.Parameter(torch.rand((dim_in, dim_in)))
            self.k_bias = nn.Parameter(torch.rand((dim_in)))
            self.v_weight = nn.Parameter(torch.rand((dim_in, dim_in)))
            self.v_bias = nn.Parameter(torch.rand((dim_in)))
            self.attn_weight = 0.25 if self.conv_branch else 1
            
            self.attn_mask = 0
            self.conv_mask = 0
            self.attn_weight = 1
            self.alpha = alpha
            self.beta = beta
            self.theta = theta
            self.drop = attn_drop
            
            # Convolution branches
            self.conv3_weight = 0.25
            self.conv5_weight = 0.25
            self.conv7_weight = 0.25
            if self.conv_branch:
                self.conv3_weight = nn.Parameter(torch.rand((dim_in, 1, 3, 3)))
                self.conv3_bias = nn.Parameter(torch.rand((dim_in)))
                
                self.conv5_weight = nn.Parameter(torch.rand((dim_in, 1, 5, 5)))
                self.conv5_bias = nn.Parameter(torch.rand((dim_in)))
                
                self.conv7_weight = nn.Parameter(torch.rand((dim_in, 1, 7, 7)))
                self.conv7_bias = nn.Parameter(torch.rand((dim_in)))
                
            # FFN
            self.ffn_in_weight = nn.Parameter(torch.rand((dim_in, dim_in)))
            self.ffn_in_bias = nn.Parameter(torch.rand((dim_in)))
            self.ffn_out_weight = nn.Parameter(torch.rand((dim_in, dim_in)))
            self.ffn_out_bias = nn.Parameter(torch.rand((dim_in)))
            self.ffn_act = nn.Sigmoid()
            
            # drop paths
            self.drop_path_1 = DropPath(drop_path) if drop_path else nn.Identity()
            self.drop_path_2 = DropPath(drop_path) if drop_path else nn.Identity()
            
            self.qkv = nn.Linear(self.dim_in, self.dim_in*3)
            self.qkv.weight.requires_grad_(False)
            self.qkv.bias.requires_grad_(False)
            self.ffn_out = nn.Linear(self.dim_in, self.dim_in)
            self.ffn_out.weight.requires_grad_(False)
            self.ffn_out.bias.requires_grad_(False)
        else:
            assert dim_out % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.num_head = dim_out // dim_head
            self.n1_input = H // self.window_size
            self.n2_input = W // self.window_size
            self.n1_output = H // self.window_size // 2
            self.n2_output = W // self.window_size // 2
            self.window_input = window_input
            # FFN with convolution
            self.ffn_in = nn.Conv2d(in_channels=dim_in,
                                    out_channels=dim_in,
                                    kernel_size=1,
                                    stride=1)
            self.conv_local = nn.Conv2d(in_channels=dim_in,
                                        out_channels=dim_out,
                                        kernel_size=dw_ks,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=math.ceil((dw_ks-stride)/2),
                                        groups=dim_in,
                                        bias=qkv_bias)
            self.ffn_out = nn.Conv2d(in_channels=dim_out,
                                     out_channels=dim_out,
                                     kernel_size=1,
                                     stride=1)
            self.post_norm = nn.LayerNorm(dim_head, elementwise_affine=False)

    def forward(self, x):
        if True:#self.training:
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                # Shortcut 1    
                q_weight_normalized, q_bias_normalized, k_weight_normalized, k_bias_normalized, v_weight_normalized, v_bias_normalized, ffn_in_weight_normalized, ffn_in_bias_normalized, ffn_out_weight_normalized, ffn_out_bias_normalized = self.stadardization()
                
                shortcut = rearrange(x, 'b n (nh hc) -> (b nh) n hc', nh=self.num_head) # B*nh, N, C/nh
                
                # Calculate Query (Q), Key (K) and Value (V)
                q = torch.nn.functional.linear(x, q_weight_normalized, q_bias_normalized) # B, N, C
                k = torch.nn.functional.linear(x, k_weight_normalized, k_bias_normalized) # B, N, C
                v = torch.nn.functional.linear(x, v_weight_normalized, v_bias_normalized) # B, N, C
                
                # Reshape
                q = rearrange(q, 'b n (nh hc) -> (b nh) n hc', nh=self.num_head) # B*nh, N, C//nh
                k = rearrange(k, 'b n (nh hc) -> (b nh) hc n', nh=self.num_head) # B*nh, C//nh, N
                v = rearrange(v, 'b n (nh hc) -> (b nh) n hc', nh=self.num_head) # B*nh, N, C//nh
                
                # Add shortcut to V
                v = v * 0.1 + shortcut * 0.95 # B*nh, N, C//nh
                    
                # Shortcut 2
                shortcut = rearrange(v, '(b nh) n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
                
                # Calculate Attention (A) and attended X
                attn = torch.bmm(q, k*self.scale).softmax(dim=-1) # B*nh, N, N
                x_spa = torch.bmm(attn, v) # B*nh, N, C//nh
                
                # Reshape x_spa
                x_spa = rearrange(x_spa, '(b nh) n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
                
                # Multiple branches during training
                if self.conv_branch:
                    x_conv = rearrange(v, '(b nh) (h w) hc -> b (nh hc) h w', nh=self.num_head, h=self.window_size, w=self.window_size) # B, C, h, w
                    # Depth-wise convolutions
                    x_conv3 = torch.nn.functional.conv2d(input = x_conv, 
                                                         weight = self.conv3_weight_orthogonal, 
                                                         bias = self.conv3_bias_zerocentric, 
                                                         stride = 1, 
                                                         padding = 'same',
                                                         groups = self.dim_in) # B, C, h, w
                    
                    x_conv5 = torch.nn.functional.conv2d(input = x_conv, 
                                                         weight = self.conv5_weight_orthogonal, 
                                                         bias = self.conv5_bias_zerocentric, 
                                                         stride = 1, 
                                                         padding = 'same',
                                                         groups = self.dim_in) # B, C, h, w
                    
                    x_conv7 = torch.nn.functional.conv2d(input = x_conv, 
                                                         weight = self.conv7_weight_orthogonal, 
                                                         bias = self.conv7_bias_zerocentric, 
                                                         stride = 1, 
                                                         padding = 'same',
                                                         groups = self.dim_in) # B, C, h, w
                else:
                    x_conv3 = 0
                    x_conv5 = 0
                    x_conv7 = 0
                
                # Fuse the outputs
                x_spa = x_spa * self.attn_weight + \
                        x_conv3 * self.conv3_weight + \
                        x_conv5 * self.conv5_weight + \
                        x_conv7 * self.conv7_weight
                
                x = self.drop_path_1(x_spa) * 0.1 +  shortcut * 0.95 # B, N, C
                
                # FFN
                # Shortcut 3
                shortcut = x # B, N, C
                x = torch.nn.functional.linear(x, ffn_in_weight_normalized, ffn_in_bias_normalized)
                x = x * 0.1 + shortcut * 0.95
                
                shortcut = x # B, N, C
                x = self.ffn_act(x) - 0.5
                x = x * 0.1 + shortcut * 0.95
                
                shortcut = x # B, N, C
                x = torch.nn.functional.linear(x, ffn_out_weight_normalized, ffn_out_bias_normalized)
                x = self.drop_path_2(x) * 0.1 + shortcut * 0.95
                
                #if x.get_device() == 0 and self.v_weight.grad is not None:
                #    print(self.ffn_out_weight.requires_grad, self.ffn_out_weight.grad)
                #    print("v:", self.v_weight.grad.mean(), self.v_weight.grad.max(), self.v_weight.grad.min())
                #    print("x:", x.var(-1).mean(), x.mean(-1).mean(), x.max(), x.min())
                #    print(self.ffn_out_weight.grad.mean(), self.ffn_out_weight.grad.max(), self.ffn_out_weight.grad.min())
                #    print(self.qkv.weight.grad.mean(), self.qkv.weight.grad.max(), self.qkv.weight.grad.min())
                #    print(x.var(-1), x.mean(-1), x.max(-1), x.min(-1))
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
                    
                x = rearrange(x, 'b (nh hc) (h n1) (w n2) -> (b n1 n2) (h w) nh hc', nh=self.num_head, n1=self.n1_output, n2=self.n2_output) 
                x = self.post_norm(x)
                x = rearrange(x, 'b n nh hc -> b n (nh hc)')
                return x  
        else:
            # Case 1: Normal multi-branch layer
            if self.attn_s:
                qkv = rearrange(self.qkv(x), 'b n (t nh hc) -> t (b nh) n hc', t=3, nh=self.num_head) # B*nh, N, C//nh
                
                attn = torch.bmm(qkv[0], (qkv[1]*self.scale).transpose(-1,-2)).softmax(dim=-1).mul_(self.attn_weight).add_(self.attn_mask) # B*nh, N, N
                x = rearrange(torch.bmm(attn, qkv[2]), '(b nh) n hc -> b n (nh hc)', nh=self.num_head) # B, N, C
                
                shortcut = x # B, N, C
                x = self.ffn_act(x) * 0.1 - 0.0066 + shortcut
                
                x = self.ffn_out(x)
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
                    
                x = rearrange(x, 'b (nh hc) (h n1) (w n2) -> (b n1 n2) (h w) nh hc', nh=self.num_head, n1=self.n1_output, n2=self.n2_output) 
                x = self.post_norm(x)
                x = rearrange(x, 'b n nh hc -> b n (nh hc)') 
                return x
                
                
    def stadardization(self):
        # For Q's weights
        weight = self.q_weight
        weight = weight.reshape(self.dim_in, self.num_head, self.dim_head)
        mean = weight.mean(dim=-1, keepdim=True)
        std = weight.std(dim=-1, keepdim=True, correction=0)
        weight_normalized = (weight - mean) / (std * math.sqrt(self.dim_in))
        q_weight_normalized = weight_normalized.reshape(self.dim_in, self.dim_in)
        q_weight_normalized = q_weight_normalized.T
        
        # For Q's bias
        bias = self.q_bias
        bias = bias.reshape(self.num_head, self.dim_head)
        mean = bias.mean(dim=-1, keepdim=True)
        std = bias.std(dim=-1, keepdim=True, correction=0)
        bias_normalized = (bias - mean) / (std * math.sqrt(self.dim_in))
        q_bias_normalized = bias_normalized.reshape(self.dim_in)
        
        # For K's weights
        weight = self.k_weight
        weight = weight.reshape(self.dim_in, self.num_head, self.dim_head)
        mean = weight.mean(dim=-1, keepdim=True)
        std = weight.std(dim=-1, keepdim=True, correction=0)
        weight_normalized = (weight - mean) / (std * math.sqrt(self.dim_in))
        k_weight_normalized = weight_normalized.reshape(self.dim_in, self.dim_in)
        k_weight_normalized = k_weight_normalized.T
        
        # For K's bias
        bias = self.k_bias
        bias = bias.reshape(self.num_head, self.dim_head)
        mean = bias.mean(dim=-1, keepdim=True)
        std = bias.std(dim=-1, keepdim=True, correction=0)
        bias_normalized = (bias - mean) / (std * math.sqrt(self.dim_in))
        k_bias_normalized = bias_normalized.reshape(self.dim_in)
        
        # For V's weights
        weight = self.v_weight
        weight = weight.reshape(self.dim_in, self.num_head, self.dim_head)
        mean = weight.mean(dim=-1, keepdim=True)
        std = weight.std(dim=-1, keepdim=True, correction=0)
        weight_normalized = (weight - mean) / (std * math.sqrt(self.dim_in))
        v_weight_normalized = weight_normalized.reshape(self.dim_in, self.dim_in)
        v_weight_normalized = v_weight_normalized.T
        
        # For V's bias
        bias = self.v_bias
        bias = bias.reshape(self.num_head, self.dim_head)
        mean = bias.mean(dim=-1, keepdim=True)
        std = bias.std(dim=-1, keepdim=True, correction=0)
        bias_normalized = (bias - mean) / (std * math.sqrt(self.dim_in))
        v_bias_normalized = bias_normalized.reshape(self.dim_in)
        
        # For ffn_in's weights
        weight = self.ffn_in_weight
        weight = weight.reshape(self.dim_in, self.num_head, self.dim_head)
        mean = weight.mean(dim=-1, keepdim=True)
        std = weight.std(dim=-1, keepdim=True, correction=0)
        weight_normalized = (weight - mean) / (std * math.sqrt(self.dim_in))
        ffn_in_weight_normalized = weight_normalized.reshape(self.dim_in, self.dim_in)
        ffn_in_weight_normalized = ffn_in_weight_normalized.T
        
        # For ffn_in's bias
        bias = self.ffn_in_bias
        bias = bias.reshape(self.num_head, self.dim_head)
        mean = bias.mean(dim=-1, keepdim=True)
        std = bias.std(dim=-1, keepdim=True, correction=0)
        bias_normalized = (bias - mean) / (std * math.sqrt(self.dim_in))
        ffn_in_bias_normalized = bias_normalized.reshape(self.dim_in)
        
        # For ffn_out's weights
        weight = self.ffn_out_weight
        weight = weight.reshape(self.dim_in, self.num_head, self.dim_head)
        mean = weight.mean(dim=-1, keepdim=True)
        std = weight.std(dim=-1, keepdim=True, correction=0)
        weight_normalized = (weight - mean) / (std * math.sqrt(self.dim_in))
        ffn_out_weight_normalized = weight_normalized.reshape(self.dim_in, self.dim_in)
        ffn_out_weight_normalized = ffn_out_weight_normalized.T
        
        # For ffn_out's bias
        bias = self.ffn_out_bias
        bias = bias.reshape(self.num_head, self.dim_head)
        mean = bias.mean(dim=-1, keepdim=True)
        std = bias.std(dim=-1, keepdim=True, correction=0)
        bias_normalized = (bias - mean) / (std * math.sqrt(self.dim_in)) # shift to zero
        ffn_out_bias_normalized = bias_normalized.reshape(self.dim_in)
                 
        return q_weight_normalized, q_bias_normalized, k_weight_normalized, k_bias_normalized, v_weight_normalized, v_bias_normalized, ffn_in_weight_normalized, ffn_in_bias_normalized, ffn_out_weight_normalized, ffn_out_bias_normalized
        
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
            """
            self.ffn_out.weight.requires_grad_(False)
            self.ffn_out.weight = self.ffn_out.weight * self.theta
            self.ffn_out.bias.requires_grad_(False)
            self.ffn_out.bias = self.ffn_out.bias * self.theta
            """

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
        
        self.norm = nn.LayerNorm(embed_dims[-1])
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
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if m.weight is not None:
                nn.init.ones_(m.weight)
        elif isinstance(m, (nn.Parameter)):
            trunc_normal_(m.data, std=.02)
            
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
        x = self.norm(x)
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
def FastAllSelfAttention_6M_767M_SingleBranch(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, False, False], conv_local=False,
                **kwargs)
    return model
    

@MODEL.register_module
def FastAllSelfAttention_6M_767M_4BranchInStage4(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, False, True], conv_local=False,
                **kwargs)
    return model
    
    
@MODEL.register_module
def FastAllSelfAttention_6M_767M_4BranchInStage34(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, False, True, True], conv_local=False,
                **kwargs)
    return model
    
@MODEL.register_module
def FastAllSelfAttention_6M_767M_4BranchInStage234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
                norm_layers=['ln_2d', 'ln_1d', 'ln_1d', 'ln_1d'], act_layers=['silu', 'silu', 'silu', 'silu'],
                dw_kss=[3, 3, 5, 5], window_sizes=[7, 7, 7, 7], attn_ss=[True, True, True, True],
                qkv_bias=True, attn_drop=0., drop_path=0.02, pre_dim=0,
                conv_branchs=[False, True, True, True], conv_local=False,
                **kwargs)
    return model
    
    
@MODEL.register_module
def FastAllSelfAttention_6M_767M_4BranchInStage1234(pretrained=False, **kwargs):
    model = EMO(# dim_in=3, num_classes=1000, img_size=224,
                depths=[3, 3, 13, 5], stem_dim=24, embed_dims=[48, 96, 192, 384], dim_heads=[16, 16, 32, 32],
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