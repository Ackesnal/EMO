from einops import rearrange, reduce
from timm.models.layers.activations import *
from timm.models.layers import DropPath, trunc_normal_, create_attn
from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE
from model.basic_modules import get_norm, get_act, ConvNormAct, LayerScale2D, MSPatchEmb

from model import MODEL

import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

inplace = True


class iRMB(nn.Module):
    def __init__(self,
                 dim_in, 
                 dim_head,
                 exp_ratio,
                 norm_layer="ln_1d",
                 act_layer="gelu",
                 window_size=7,
                 window_num=1,
                 qkv_bias=True,
                 attn_drop=0.0,
                 drop=0.0,
                 drop_path=0.0,
                 group=1):
        
        super().__init__()
        
        assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
        self.dim_in = dim_in
        self.dim_head = dim_head
        self.dim_exp = int(dim_in*exp_ratio)
        self.num_head = self.dim_in//self.dim_head
        self.window_size = window_size
        self.window_num = window_num
        
        self.pre_norm = get_norm(norm_layer)(self.dim_in) 
        self.post_norm = get_norm(norm_layer)(self.dim_exp//2)
        
        self.scale = self.dim_head ** -0.5
        self.qk = nn.Conv2d(self.dim_in, self.dim_in*2, kernel_size=1, stride=1, bias=qkv_bias, groups=group)
        self.v = nn.Conv2d(self.dim_in, self.dim_exp, kernel_size=1, stride=1, bias=qkv_bias, groups=group)
        self.attn_drop = attn_drop
        
        self.act = nn.GELU()
        self.proj = nn.Conv2d(self.dim_exp, self.dim_in, kernel_size=1, stride=1, bias=qkv_bias)
        self.proj_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros(window_num, (2*window_size-1)**2, self.num_head))  # (2*W-1)^2, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index = relative_position_index.reshape(1, -1, 1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        
        # padding & reshape
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
        
        # pre norm
        x = self.pre_norm(x)
        
        # attention
        b, c, h, w = x.shape
        qk = self.qk(x)
        qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
        
        v = self.v(x)
        v_attn, v_idle = torch.chunk(v, chunks=2, dim=1)
        v_attn = rearrange(v_attn, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head, dim_head=self.dim_exp//2//self.num_head).contiguous()
        
        relative_position_bias = torch.gather(self.relative_position_bias_table,
                                              dim = 1,
                                              index = self.relative_position_index.expand(self.window_num, -1, self.num_head))
        relative_position_bias = relative_position_bias.view(self.window_num, 
                                                             self.window_size*self.window_size, 
                                                             self.window_size*self.window_size, 
                                                             self.num_head)
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous().unsqueeze(0).expand(B,-1,-1,-1,-1).reshape(b, self.num_head, self.window_size*self.window_size, self.window_size*self.window_size)
        
        
        q, k = qk[0], qk[1]
        v_attn = F.scaled_dot_product_attention(q, k, v_attn, attn_mask=relative_position_bias, dropout_p=self.attn_drop)
        v_attn = rearrange(v_attn, 'b heads (h w) dim_head -> b (heads dim_head) h w', h=h, w=w).contiguous()
        
        # post norm
        v_attn = self.post_norm(v_attn)
        
        # unpadding
        x = torch.cat((v_attn, v_idle), dim=1)
        x = rearrange(x, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        
        x = self.act(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = shortcut + self.drop_path(x)
        
        # rotate
        x = torch.roll(x, shifts=(2,2), dims=(2,3))
        return x


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
        self.stage0 = nn.ModuleList([MSPatchEmb(dim_in, dim_stem, kernel_size=3, c_group=1,
                                                stride=2, norm_layer="bn_2d", act_layer='silu')])
        img_size = img_size//2
                                                
        for i in range(len(depths)):
            layers = []
            dpr = dprs[sum(depths[:i]):sum(depths[:i+1])]
            for j in range(depths[i]):
                if j == 0:
                    if i == 0:
                        dim_in = dim_stem
                    else:
                        dim_in = embed_dims[i-1]
                    layers.append(MSPatchEmb(dim_in, embed_dims[i], kernel_size=3, c_group=1, dilations=[1],
                                             stride=2, norm_layer="bn_2d", act_layer='silu'))
                    img_size = img_size//2
                else:
                    layers.append(iRMB(embed_dims[i], dim_heads[i], exp_ratio=exp_ratios[i], 
                                       norm_layer=norm_layers[i], act_layer=act_layers[i],
                                       window_size=window_sizes[i], 
                                       window_num=(img_size//window_sizes[i])**2,
                                       qkv_bias=qkv_bias, attn_drop=attn_drop, 
                                       drop=drop, drop_path=dpr[j], group=group))
                
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
                x = checkpoint.checkpoint(blk, x.requires_grad_(), use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage1:
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_(), use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage2:
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_(), use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage3:
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_(), use_reentrant=False)
            else:
                x = blk(x)
        for blk in self.stage4:
            if self.training:
                x = checkpoint.checkpoint(blk, x.requires_grad_(), use_reentrant=False)
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
                depths=[2, 2, 14, 2],
                dim_stem=48,
                embed_dims=[48, 96, 192, 384],
                exp_ratios=[2, 2, 2, 2],
                norm_layers=['ln_2d', 'ln_2d', 'ln_2d', 'ln_2d'],
                act_layers=['gelu', 'gelu', 'gelu', 'gelu'],
                dim_heads=[16, 16, 16, 16],
                window_sizes=[7, 7, 7, 7],
                qkv_bias=True,
                attn_drop=0.,
                drop=0.,
                drop_path=0.01,
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
