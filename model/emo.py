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
    def __init__(self,
                 dim_in,
                 exp_ratio,
                 window_size=7,
                 dim_head=16,
                 drop=0.0,
                 drop_path=0.0,
                 shuffle=False,
                 layer_scale_init_value=1e-6,
                 reverse=False):
        
        super().__init__()
        
        self.dim_in = dim_in
        self.window_size = window_size
        
        # define a parameter table of relative position bias
        #self.relative_pos_biases = nn.Parameter(torch.zeros((dim_in, self.window_size**2, self.window_size**2)))  # (2*W-1)^2, nH
        #trunc_normal_(self.relative_pos_biases, std=.02)
        
        """
        # define a parameter table of relative position bias
        self.relative_pos_biases = nn.Parameter(torch.zeros((dim_in, (2*self.window_size-1)**2+1)))  # (2*W-1)^2, nH
        trunc_normal_(self.relative_pos_biases, std=.02)
        self.relative_pos_indices = self.generate_pos_indices(H = self.window_size, 
                                                              W = self.window_size, 
                                                              K = self.window_size*2-1) # wh*ww, wh*ww
        self.relative_pos_bias = torch.gather(self.relative_pos_biases.unsqueeze(1).expand(-1, window_size*window_size, -1),
                                              2,
                                              self.relative_pos_indices.unsqueeze(0).expand(dim_in, -1, -1))
        """
        
        self.conv3 = nn.Conv2d(dim_in, dim_in, kernel_size=3, groups=dim_in, 
                               stride=1, padding="same", dilation=1)
        self.conv5 = nn.Conv2d(dim_in, dim_in, kernel_size=5, groups=dim_in, 
                               stride=1, padding="same", dilation=1)
        self.conv7 = nn.Conv2d(dim_in, dim_in, kernel_size=7, groups=dim_in, 
                               stride=1, padding="same", dilation=1)
        self.qk = nn.Conv2d(dim_in, dim_in, kernel_size=1, stride=1)
        
        self.norm = nn.LayerNorm(dim_in)
        self.act = nn.GELU()
        self.proj1 = nn.Conv2d(dim_in, dim_in*exp_ratio, kernel_size=1, stride=1)
        self.proj2 = nn.Conv2d(dim_in*exp_ratio, dim_in, kernel_size=1, stride=1)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        
        self.gamma_conv3 = nn.Parameter(layer_scale_init_value * torch.ones((dim_in)), 
                                        requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_conv5 = nn.Parameter(layer_scale_init_value * torch.ones((dim_in)), 
                                        requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_conv7 = nn.Parameter(layer_scale_init_value * torch.ones((dim_in)),
                                        requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma_attn = nn.Parameter(layer_scale_init_value * torch.ones((dim_in)), 
                                       requires_grad=True) if layer_scale_init_value > 0 else None   
        
        self.shuffle = shuffle
        self.reverse = reverse
        self.dim_head = dim_head
        self.num_head = dim_in // self.dim_head
        
    def forward(self, x):
        """
        x: B*C*H*W
        """
        shortcut = x
        B, C, H, W = x.shape
        
        # Conver to windows
        if self.window_size <= 0:
            window_size_W, window_size_H = W, H
        else:
            window_size_W, window_size_H = self.window_size, self.window_size
        pad_l, pad_t = 0, 0
        pad_r = (window_size_W - W % window_size_W) % window_size_W
        pad_b = (window_size_H - H % window_size_H) % window_size_H
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
        n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
        x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous() # B*nh*nw, C, wh, ww
        
        b, c, h, w = x.shape
        
        # Convolutional layers
        x_conv3 = self.conv3(x) # B*nh*nw, C, wh, ww
        x_conv5 = self.conv5(x) # B*nh*nw, C, wh, ww
        x_conv7 = self.conv7(x) # B*nh*nw, C, wh, ww
        
        
        # MHSA layer
        x_qk = self.qk(x) # B*nh*nw, C, wh, ww
        x_qk = x_qk.reshape(b, c, h*w).permute(0,2,1) # B*nh*nw, wh*ww, C
        x_q, x_k = x_qk[:,:,:c//2], x_qk[:,:,c//2:] # B*nh*nw, wh*ww, C/2
        x_q = x_q.reshape(b, h*w, self.num_head, -1).transpose(1,2).contiguous() # B*nh*nw, head, wh*ww, C//2//head
        x_k = x_k.reshape(b, h*w, self.num_head, -1).transpose(1,2).contiguous() # B*nh*nw, head, wh*ww, C//2//head
        x_v = x.reshape(b, self.num_head, c//self.num_head, h*w).transpose(2,3).contiguous() # B*nh*nw, head, wh*ww, C//head
        
        # self.relative_pos_bias = self.relative_pos_bias.unsqueeze(0).expand(b, -1, -1, -1) # B*nh*nw, C, wh*ww, wh*ww
        x_attn = F.scaled_dot_product_attention(x_q, x_k, x_v) # B*nh*nw, head, wh*ww, C//head
        x_attn = x_attn.transpose(2,3).reshape(b, c, h, w)
        
        
        # concatenate
        x = x_conv3 * self.gamma_conv3.reshape(1,-1,1,1) + \
            x_conv5 * self.gamma_conv5.reshape(1,-1,1,1) + \
            x_conv7 * self.gamma_conv7.reshape(1,-1,1,1) + \
            x_attn * self.gamma_attn.reshape(1,-1,1,1)
        
        # unpadding
        x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        
        x = shortcut + self.drop_path(x)
        
        shortcut = x
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        
        x = shortcut + self.drop_path(x)
        
        if self.shuffle:
            # rotate
            if self.reverse:
                x = x.reshape(B, C, H//2, 2, W//2, 2).permute(0,1,3,2,5,4).reshape(B,C,H,W)
            else:
                x = x.reshape(B, C, 2, H//2, 2, W//2).permute(0,1,3,2,5,4).reshape(B,C,H,W)    
        return x                     
        
    def generate_pos_indices(self, H, W, K):
        """
        Generate the indices for testing phase
        """
        # Total number of pixels
        N = H * W
        
        # Create a tensor representing the 2D coordinates of each pixel in the image
        coords = torch.stack(torch.meshgrid(torch.arange(H+1), torch.arange(W+1)), dim=-1)
        
        # Compute the relative positions for each pixel with its KxK neighborhood
        half_K = K // 2
        relative_coords = torch.stack([coords + torch.tensor([[i - half_K, j - half_K]]) for i in range(K) for j in range(K)], dim=2)
        
        # Handle boundary conditions by clamping the coordinates
        relative_coords[..., 0].clamp_(0, H)
        relative_coords[..., 1].clamp_(0, W)
        
        # Convert the 2D relative coordinates to a single index
        indices = relative_coords[..., 0] * (W+1) + relative_coords[..., 1]
    
        # Initialize the matrix with zeros
        matrix = torch.zeros(H*W, (H+1)*(W+1), dtype=torch.int64)
        
        # Create a relationship label tensor
        relationship_labels = torch.arange(1, K*K + 1).reshape(-1)
    
        # Fill the matrix with the relationship labels
        for i in range(H):
            for j in range(W):
                row_idx = i * W + j
                matrix[row_idx, indices[i, j].reshape(-1)] = relationship_labels
        
        matrix = matrix.reshape(H*W, H+1, W+1)[:, :-1, :-1].reshape(H*W, H*W)
        return matrix


class EMO(nn.Module):
    def __init__(self, 
                 dim_in=3,
                 num_classes=1000,
                 img_size=224,
                 depths=[1, 2, 4, 2],
                 dim_stem=16,
                 embed_dims=[64, 128, 256, 512],
                 exp_ratios=[4., 4., 4., 4.],
                 norm_layers=['bn_2d', 'bn_2d', 'bn_2d', 'bn_2d'],
                 act_layers=['relu', 'relu', 'relu', 'relu'],
                 dim_heads=[32, 32, 32, 32],
                 window_sizes=[7, 7, 7, 7],
                 qkv_bias=True,
                 attn_drop=0.,
                 drop=0.,
                 drop_path=0.,
                 group=1):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes > 0
        dprs = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList([MSPatchEmb(dim_in, dim_stem, kernel_size=3, c_group=-1,
                                                stride=2, norm_layer="ln_2d", act_layer='gelu')])
        img_size = img_size//2
                                                
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i+1])]
            if i == 0:
                dim_in = dim_stem
            else:
                dim_in = embed_dims[i-1]
            
            # Downsampling
            layers.append(MSPatchEmb(dim_in, embed_dims[i], kernel_size=3, c_group=-1,
                                     stride=2, norm_layer="ln_2d", act_layer='gelu'))
            img_size = img_size//2
            
            # Integrated MHSA
            for j in range(depths[i]):
                layers.append(iRMB(embed_dims[i], exp_ratio=exp_ratios[i],
                                   window_size=window_sizes[i], dim_head=dim_heads[i],
                                   drop=drop, drop_path=dpr[j], 
                                   shuffle=True if i < len(depths)-1 and j <=depths[i]//2*2 else False,
                                   reverse=True if j%2==1 else False))
                
            self.__setattr__(f'stage{i + 1}', nn.ModuleList(layers))
    		
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.apply(self._init_weights)
	
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
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
                x = checkpoint.checkpoint(blk, x.requires_grad_())
            else:
                x = blk(x)
        for i, blk in enumerate(self.stage1):
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_())
            else:
                x = blk(x)
        for i, blk in enumerate(self.stage2):
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_())
            else:
                x = blk(x)
        for i, blk in enumerate(self.stage3):
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_())
            else:
                x = blk(x)
        for i, blk in enumerate(self.stage4):
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_())
            else:
                x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = reduce(x, 'b c h w -> b c', 'mean').contiguous()
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
	model = EMO(
		# dim_in=3, num_classes=1000, img_size=224,
		depths=[3, 3, 9, 3], stem_dim=24, embed_dims=[48, 72, 160, 320], exp_ratios=[2., 3., 4., 5.],
		norm_layers=['bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'], act_layers=['silu', 'silu', 'gelu', 'gelu'],
		dw_kss=[3, 3, 5, 5], dim_heads=[16, 24, 20, 32], window_sizes=[7, 7, 7, 7], attn_ss=[False, False, True, True],
		qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05, v_group=False, attn_pre=True, pre_dim=0,
		**kwargs)
	return model
 
 
@MODEL.register_module
def Shufformer_6M(pretrained=False, **kwargs):
    model = EMO(dim_in=3,
                img_size=224,
                depths=[3, 3, 11, 3],
                dim_stem=48,
                embed_dims=[48, 96, 192, 384],
                exp_ratios=[3, 3, 3, 3],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'],
                act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dim_heads=[16, 32, 32, 64],
                window_sizes=[14, 14, 14, 7],
                qkv_bias=True,
                attn_drop=0.,
                drop=0.,
                drop_path=0.1,
                group=1,
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
