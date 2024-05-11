import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from torch import nn
from utils.eval import count_parameters

from models.backbone.Swin_Transformer import swin_backbone, PatchMerging, Mlp,swin_backbone_tiny
from models.backbone.Vision_Transformer import vit_backbone


class MultiPartsSampling(nn.Module):
	def __init__(self, dim, input_size, backbone=None, parts_ratio=2, num_heads=16,
	            feature_weights_pooling=True, att_drop=0.2, head_drop=0.5,
	             parts_drop=0.2, num_classes=200, pos=True, parts_base=0., cross_layer=False,
	             label_smooth=0.0, mixup=0.,backbone_type='hier'):
		super(MultiPartsSampling, self).__init__()
		self.num_heads = num_heads
		self.parts_ratio = parts_ratio
		self.input_size = (input_size//32,input_size//32) if backbone_type=='hier' else (input_size//16,input_size//16)
		self.dim = dim
		self.num_classes = num_classes
		self.fwp = feature_weights_pooling
		self.mixup = mixup
		self.ce = LabelSmoothingCrossEntropy(smoothing=label_smooth)
		self.ls = nn.LogSoftmax(dim=-1)
		self.activation = nn.GELU()
		self.dis = nn.KLDivLoss(reduction='batchmean',log_target=True)
		self.dis = nn.PairwiseDistance(2,1e-8)
		self.block = MultiPartRetrospect(self.dim, self.input_size, parts_ratio, self.num_heads,
		                                 att_drop, parts_drop, pos, parts_base, cross_layer, backbone_type)
		self.backbone_type = backbone_type
		if self.fwp:
			self.pooling = self.feature_weights_pooling
			stage_weights = 4 if cross_layer else 1
			self.pooling_weights = nn.Parameter(torch.ones(stage_weights, 1, 1, 1) / stage_weights)
		else:
			self.pooling = nn.AdaptiveAvgPool1d(1)
			# self.pooling2 = nn.AdaptiveMaxPool1d(1)
			self.conv = nn.Conv2d(2,1,3,1,1)
		if cross_layer:
			new_dim = int(dim * 15 / 8) if self.backbone_type == 'hier' else int(dim * 4)
			self.norm = nn.LayerNorm(new_dim)
			self.head = nn.Linear(new_dim, num_classes)
		else:
			self.norm = nn.LayerNorm(dim)
			self.head = nn.Linear(dim, num_classes)
		self.head_drop = nn.Dropout(head_drop)
		self.show = nn.Identity()
		self.apply(self.init_weights)
		self.backbone = backbone
		self.assess = False
		self.save_feature = None
		self.count = 0

	def feature_weights_pooling(self, x, feature_weights):
		# feature_weights = (feature_weights + self.pooling_weights).sum(0)
		if self.assess:
			os.makedirs(f'../visualize/feature_weights/', exist_ok=True)
			torch.save(x, '../visualize/feature_weights/x.pt')
			torch.save(feature_weights, '../visualize/feature_weights/weights.pt')
		sum_feature_weights = (feature_weights * self.pooling_weights).sum(0)
		sum_feature_weights = sum_feature_weights / sum_feature_weights.sum(-2).unsqueeze(-1)
		x = x @ sum_feature_weights
		return x, sum_feature_weights

	def init_weights(self, m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			nn.init.kaiming_normal_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		nn.init.constant_(self.head.weight, 0)
		nn.init.constant_(self.head.bias, 0)

	def normalize_cam(self,grad):
		min,_ = grad.min(-1, keepdim=True)
		grad = grad - min
		grad = grad / (1e-8 + grad.sum(-1, keepdim=True))
		return grad

	def flops(self):
		flops = 0
		# Backbone
		flops += self.backbone.flops()
		# HPR
		flops += self.block.flops()
		# Delete Original Norm
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Delete Original Head
		flops -= self.dim * self.input_size[0] * self.input_size[0]
		# Norm
		flops += self.dim * 15 / 8 * self.input_size[0] * self.input_size[0]
		# Multi-Grained Fusion
		flops += self.dim * self.input_size[0] * self.input_size[0]
		# Head
		flops += self.dim * 15 / 8 * self.num_classes
		return flops


	def forward(self, x, label=None):
		x = self.backbone(x)
		x, feature_weights = self.block(x)
		featmap = x
		x = self.norm(x)
		x = self.head_drop(x)
		if self.fwp:
			x, sum_feature_weights = self.pooling(x.transpose(-2, -1), feature_weights)
			x = x.flatten(1)
			sum_feature_weights = sum_feature_weights.squeeze(-1)
		else:
			# CBAM
			# x1 = x.mean(-1)
			# x2,_ = x.max(-1)
			# sa = torch.stack((x1,x2),-1)
			# sa = format_reverse(sa)
			# sa = self.conv(sa)
			# sa = format_reverse(sa)
			# x = x * sa

			x = self.pooling(x.transpose(-2, -1)).flatten(1)

		# if self.save_feature:
		self.save_feature = x

		x = self.head(x)
		if self.training and label is not None and self.mixup <= 0:
			loss_ce, loss_cam=0, 0
			loss_ce = self.ce(x, label)
			if self.fwp:
				grad = torch.autograd.grad(loss_ce, featmap, torch.ones_like(loss_ce), True)[0].detach()
				gradcam = self.activation((featmap.detach() * grad.mean(1, keepdim=True)).sum(dim=-1))
				gradcam = self.normalize_cam(gradcam).detach()
				if self.assess:
					os.makedirs(f'../visualize/feature_weights/', exist_ok=True)
					torch.save(gradcam, '../visualize/feature_weights/gradcam.pt')
				# loss_cam = self.dis(self.ls(sum_feature_weights), self.ls(gradcam)).mean()
				loss_cam = self.dis(sum_feature_weights, gradcam).mean()
			loss = [loss_ce + loss_cam, loss_ce, loss_cam]
			return x, loss
		else:
			return x


class MultiPartRetrospect(nn.Module):
	def __init__(self, dim, input_size, parts_ratio=1, num_heads=4, att_drop=0.2, parts_drop=0.2,
	             pos=True, parts_base=0., cross_layer=False, backbone_type='hier'):
		super().__init__()
		self.input_size = input_size
		self.cross_layer = cross_layer
		self.num_parts = dim // parts_ratio
		if self.cross_layer:
			self.norm_list = nn.ModuleList()
			self.parts_generation_list = nn.ModuleList()
			self.mpsa_list = nn.ModuleList()
			stage_scale_list = [8, 4, 2, 1] if backbone_type == 'hier' else [1,1,1,1]
			num_parts_list = [dim // (stage_scale_list[0] * parts_ratio),
			                  dim // (stage_scale_list[1] * parts_ratio),
			                  dim // (stage_scale_list[2] * parts_ratio),
			                  dim // (stage_scale_list[3] * parts_ratio)]

			for stage, parts in zip(stage_scale_list, num_parts_list):
				self.norm_list.append(nn.LayerNorm(dim // stage))
				self.parts_generation_list.append(
					PartSampling(dim // stage, (input_size[0] * stage, input_size[1] * stage),
					             parts, pos))
				self.mpsa_list.append(PartSamplingAttention(dim // stage, dim, input_size, parts,
				                                            dim // num_heads, parts_base,
				                                            att_drop, parts_drop))

		else:
			self.norm = nn.LayerNorm(dim)
			self.parts_generation = PartSampling(dim, input_size, self.num_parts, pos)
			self.mpsa = PartSamplingAttention(dim, dim, input_size, self.num_parts, dim // num_heads, parts_base)
		self.activation = nn.GELU()


	def flops(self):
		flops = 0
		if self.cross_layer:
			for norm, part_generation, mpsa in zip(self.norm_list, self.parts_generation_list, self.mpsa_list):
				flops += part_generation.flops()
				flops += mpsa.flops()
		return flops


	def forward(self, x):
		if self.cross_layer:
			out_list, feature_weights_list = [], []
			for i in range(4):
				x[i] = self.norm_list[i](x[i])
			for i in range(4):
				parts = self.parts_generation_list[i](x[i])
				out, feature_weights = self.mpsa_list[i](x[-1], parts)
				out_list.append(out)
				feature_weights_list.append(feature_weights)
			out = torch.cat(out_list, dim=-1)
			feature_weights = torch.stack(feature_weights_list, dim=0)
			x = self.activation(out)
		else:
			x = x[-1]
			x = self.norm(x)
			parts = self.parts_generation(x)
			out, feature_weights = self.mpsa(x, parts)
			x = self.activation(out)
		return x, feature_weights


class PartSamplingAttention(nn.Module):
	def __init__(self, dim, query_dim, query_size, num_parts=512, heads_dim=4, parts_base=0.,
	             att_drop=0.2, parts_drop=0.2):
		super().__init__()
		self.q = nn.Linear(query_dim, dim)
		self.kv = nn.Linear(dim, dim * 2)
		self.num_parts = num_parts
		self.num_heads = dim // heads_dim
		self.query_dim = query_dim
		self.parts_base = parts_base
		self.parts_drop = int(self.num_parts * parts_drop)
		self.dim = dim
		if self.parts_base:
			self.parts_base_num = int(self.num_parts * self.parts_base)
			self.learnable_parts = nn.Parameter(torch.randn((1, self.parts_base_num, dim)))
			self.num_parts = self.num_parts + self.parts_base_num
			torch.nn.init.kaiming_normal_(self.learnable_parts)
		self.parts_attention = PartSE(self.num_parts - self.parts_drop)
		self.scale = heads_dim ** -0.5
		self.atten_pos = nn.Parameter(torch.zeros((1, self.num_heads, query_size[0] * query_size[1],
		                                           self.num_parts - self.parts_drop)))
		self.softmax = nn.Softmax(dim=-1)
		self.softmax2 = nn.Softmax(dim=-2)
		self.weights_scale = nn.Parameter(torch.tensor(0.1))
		self.o = nn.Linear(dim, dim)
		self.dropout = nn.Dropout(att_drop)
		self.attn_drop = att_drop

	def flops(self):
		flops = 0
		# Q projection
		flops += self.num_tokens * self.dim * self.dim
		# KV projection
		flops += self.parts_drop * self.query_dim * self.dim
		# Parts Attention
		flops += self.parts_attention.flops()
		# Attention Map
		flops += self.parts_drop * self.dim * self.num_tokens
		# V
		flops += self.parts_drop * self.num_tokens * self.dim
		# Add Learnable Bias
		flops += self.num_heads * self.num_parts * self.num_tokens
		# Feature Weights
		flops += self.parts_drop * self.num_tokens
		# Enhance Feature Map
		flops += self.num_tokens * self.dim
		# O
		flops += self.dim * self.dim * self.num_tokens
		return flops


	def forward(self, x, parts):
		B, N, C = x.shape
		self.num_tokens = N
		if self.parts_base:
			learnable_parts = self.learnable_parts.expand(B, -1, -1)
			parts = torch.cat((parts, learnable_parts), dim=1)

		if self.parts_drop:
			parts_index = torch.randperm(self.num_parts)
			remained = parts_index[self.parts_drop:]
			parts = parts[:, remained]
		# atten_pos = self.atten_pos[:, :, :, remained]
		# parts_attention = parts_attention[:,:,:, remained]
		# print(parts.shape)

		q = rearrange(self.q(x), 'b n (h hc) -> b h n hc', h=self.num_heads)
		kv = rearrange(self.kv(parts), 'b p (kv h hc) -> kv b h p hc', kv=2, h=self.num_heads)
		k, v = kv[0], kv[1]

		parts_attention = self.parts_attention(parts)
		attention_weights = (q @ k.transpose(-2, -1).contiguous()) * self.scale
		attention_weights = attention_weights + self.atten_pos

		# # Drop Key
		# if self.training:
		# 	m_r = torch.ones_like(attention_weights) * self.attn_drop
		# 	attention_weights = attention_weights + torch.bernoulli(m_r) * -1e12

		attention_score = self.softmax(attention_weights)


		# Drop Attention
		attention_score = self.dropout(attention_score)

		x = rearrange(attention_score @ v, 'b h n hc -> b n (h hc)')
		# x = rearrange(attention_score @ (v+self.weights_scale*parts_attention.reshape(B,1,-1,1)), 'b h n hc -> b n (h hc)')
		feature_weights = self.softmax2(attention_weights).mean(1) @ parts_attention.reshape(B, -1, 1)

		feature_weights = (feature_weights - feature_weights.min()) / (feature_weights.max() - feature_weights.min())
		x = x + feature_weights * self.weights_scale
		x = self.o(x)

		return x, feature_weights


class PartSE(nn.Module):
	def __init__(self, num_parts):
		super().__init__()
		self.gap = nn.AdaptiveAvgPool1d(1)
		self.conv1 = nn.Conv2d(num_parts, num_parts, 1)
		self.conv2 = nn.Conv2d(num_parts, num_parts, 1)
		self.ln = nn.LayerNorm([num_parts, 1, 1])
		self.activation = nn.GELU()
		self.standardize = nn.Sigmoid()

	def flops(self):
		flops = 0
		# GAP
		flops += self.Cr * self.C
		# Conv1
		flops += self.Cr * self.Cr
		# LN + Act + Sigmoid
		flops += 2 * self.Cr
		# Conv2
		flops += self.Cr * self.Cr
		return flops



	def forward(self, x):
		B, Cr, C = x.shape
		self.Cr,self.C = Cr, C
		x = self.gap(x).unsqueeze(-1)
		x = self.conv1(x)
		x = self.ln(x)
		x = self.activation(x)
		x = self.conv2(x)
		x = self.standardize(x).reshape(B, 1, 1, Cr)  # (B,1,1,C/r)
		return x


class PartSampling(nn.Module):
	def __init__(self, dim, input_size, num_parts, pos=True, ):
		super().__init__()
		self.pos = pos
		self.linear = nn.Linear(dim, num_parts)
		self.dim = dim
		self.num_parts = num_parts
		self.part_pos = nn.Parameter(torch.zeros((1, input_size[0] * input_size[1], num_parts)))
		self.softmax = nn.Softmax(dim=-1)
		self.activation = nn.GELU()
		self.assess = False


	def flops(self):
		flops = 0
		# Projection
		flops += self.num_parts * self.dim * self.dim
		# Part Features
		flops += self.num_parts * self.num_tokens * self.dim
		return flops


	def forward(self, x):
		B,N,C = x.shape
		self.num_tokens = N
		parts = self.linear(x)
		parts = self.activation(parts)
		if self.pos:
			parts = parts + self.part_pos
		sample_map = self.softmax(rearrange(parts, 'b hw cr -> b cr hw'))
		x = sample_map @ x  # (B,C/r,C)

		if self.assess:
			os.makedirs(f'../visualize/sampling_map/map_file/', exist_ok=True)
			if sample_map.shape[-1] == 9216:
				torch.save(sample_map, '../visualize/sampling_map/map_file/part_sample_1.pt')
			elif sample_map.shape[-1] == 2304:
				torch.save(sample_map, '../visualize/sampling_map/map_file/part_sample_2.pt')
			elif sample_map.shape[-1] == 576:
				torch.save(sample_map, '../visualize/sampling_map/map_file/part_sample_3.pt')
			elif sample_map.shape[-1] == 144:
				torch.save(sample_map, '../visualize/sampling_map/map_file/part_sample_4.pt')
		# print('Do not save the Sample Map')
		return x


def format_reverse(x):
	if isinstance(x, (list, tuple)):
		conv_in = False if len(x[0].shape) == 3 else True
		if conv_in:
			res = [rearrange(x0, 'b c h w -> b (h w) c') for x0 in x]
		else:
			print(x[0].shape)
			h = int(math.sqrt(x[0].shape[-2]))
			res = [rearrange(x0, 'b (h w) c -> b c h w', h=h) for x0 in x]
	else:
		conv_in = False if len(x.shape) == 3 else True
		if conv_in:
			res = rearrange(x, 'b c h w -> b (h w) c')
		else:
			h = int(math.sqrt(x[0].shape[-2]))
			res = rearrange(x, 'b (h w) c -> b c h w', h=h)
	return res


if __name__ == '__main__':
	dim = 1024
	img_size = 384
	batch = 2
	x = torch.rand(batch, 3, img_size, img_size)
	label = torch.randint(200,(batch,))
	# pretrained = "D:\\swin_base_patch4_window12_384_22k.pth"
	backbone = swin_backbone(window_size=img_size//32, img_size=img_size, num_classes=200, cross_layer=True)
	# model = swin_backbone(window_size=12, img_size=384, num_classes=200, cross_layer=True)
	# backbone = vit_backbone()
	# y = backbone(x)
	mps = MultiPartsSampling(dim, img_size, backbone,
	                         parts_drop=0.2, parts_base=0., cross_layer=True, feature_weights_pooling=False)

	from thop import profile

	print('Backbone FLOPs = ' + str(mps.backbone.flops() / 1000 ** 3) + 'G')
	print('Backbone Params = ' + str(count_parameters(mps.backbone)) + 'M')

	y, loss = mps(x, label)

	print('Ours FLOPs = ' + str(mps.flops() / 1000 ** 3) + 'G')
	print('Ours Params = ' + str(count_parameters(mps)) + 'M')


	print(loss)
# print(mps)
# ipa = InterPartsAttention(1024).cuda()
# x = torch.rand(2,256,1024,device='cuda')
# y = ipa(x)
# print(y.shape)
