import math
from functools import partial

import torch
import torch.nn.functional as F
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropBlock2d, DropPath
from torch import nn
from einops import rearrange
from models.mps import format_reverse


def get_padding(kernel_size, stride, dilation=1):
	padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
	return padding


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(
			self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
			reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
			aa_layer=None, drop_block=None, drop_path=None):
		super(Bottleneck, self).__init__()

		width = int(math.floor(planes * (base_width / 64)) * cardinality)
		first_planes = width // reduce_first
		outplanes = planes * self.expansion
		first_dilation = first_dilation or dilation
		use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

		self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
		self.bn1 = norm_layer(first_planes)
		self.act1 = act_layer(inplace=True)

		self.conv2 = nn.Conv2d(
			first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
			padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
		self.bn2 = norm_layer(width)
		self.drop_block = drop_block() if drop_block is not None else nn.Identity()
		self.act2 = act_layer(inplace=True)
		self.aa = nn.Identity()

		self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
		self.bn3 = norm_layer(outplanes)

		self.act3 = act_layer(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.dilation = dilation
		self.drop_path = drop_path

	def zero_init_last(self):
		nn.init.zeros_(self.bn3.weight)

	def forward(self, x):
		shortcut = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act1(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.drop_block(x)
		x = self.act2(x)
		x = self.aa(x)

		x = self.conv3(x)
		x = self.bn3(x)

		if self.drop_path is not None:
			x = self.drop_path(x)

		if self.downsample is not None:
			shortcut = self.downsample(shortcut)
		x += shortcut
		x = self.act3(x)

		return x


def downsample_conv(
		in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
	norm_layer = norm_layer or nn.BatchNorm2d
	kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
	first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
	p = get_padding(kernel_size, stride, first_dilation)

	return nn.Sequential(*[
		nn.Conv2d(
			in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
		norm_layer(out_channels)
	])


def downsample_avg(
		in_channels, out_channels, kernel_size, stride=1, dilation=1, first_dilation=None, norm_layer=None):
	norm_layer = norm_layer or nn.BatchNorm2d
	avg_stride = stride if dilation == 1 else 1
	if stride == 1 and dilation == 1:
		pool = nn.Identity()
	else:
		pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)

	return nn.Sequential(*[
		pool,
		nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
		norm_layer(out_channels)
	])


def drop_blocks(drop_prob=0.):
	return [
		None, None,
		partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
		partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
		block_fn, channels, block_repeats, inplanes, reduce_first=1, output_stride=32,
		down_kernel_size=1, avg_down=False, drop_block_rate=0., drop_path_rate=0., **kwargs):
	stages = []
	feature_info = []
	net_num_blocks = sum(block_repeats)
	net_block_idx = 0
	net_stride = 4
	dilation = prev_dilation = 1
	for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
		stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
		stride = 1 if stage_idx == 0 else 2
		if net_stride >= output_stride:
			dilation *= stride
			stride = 1
		else:
			net_stride *= stride

		downsample = None
		if stride != 1 or inplanes != planes * block_fn.expansion:
			down_kwargs = dict(
				in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
				stride=stride, dilation=dilation, first_dilation=prev_dilation, norm_layer=kwargs.get('norm_layer'))
			downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

		block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
		blocks = []
		for block_idx in range(num_blocks):
			downsample = downsample if block_idx == 0 else None
			stride = stride if block_idx == 0 else 1
			block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
			blocks.append(block_fn(
				inplanes, planes, stride, downsample, first_dilation=prev_dilation,
				drop_path=DropPath(block_dpr) if block_dpr > 0. else None, **block_kwargs))
			prev_dilation = dilation
			inplanes = planes * block_fn.expansion
			net_block_idx += 1

		stages.append((stage_name, nn.Sequential(*blocks)))
		feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

	return stages, feature_info


class ResNet(nn.Module):

	def __init__(
			self, block, layers, num_classes=1000, in_chans=3, output_stride=32, global_pool='avg',
			cardinality=1, base_width=64, block_reduce_first=1, down_kernel_size=1, avg_down=False, act_layer=nn.ReLU,
			norm_layer=nn.BatchNorm2d, aa_layer=None,
			drop_rate=0.0, drop_path_rate=0., drop_block_rate=0., zero_init_last=True, block_args=None):
		super(ResNet, self).__init__()
		block_args = block_args or dict()
		assert output_stride in (8, 16, 32)
		self.num_classes = num_classes
		self.drop_rate = drop_rate
		self.grad_checkpointing = False

		# Stem
		inplanes = 64
		self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(inplanes)
		self.act1 = act_layer(inplace=True)
		self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

		# Stem pooling. The name 'maxpool' remains for weight compatibility.
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# Feature Blocks
		channels = [64, 128, 256, 512]
		stage_modules, stage_feature_info = make_blocks(
			block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width,
			output_stride=output_stride, reduce_first=block_reduce_first, avg_down=avg_down,
			down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, aa_layer=aa_layer,
			drop_block_rate=drop_block_rate, drop_path_rate=drop_path_rate, **block_args)
		for stage in stage_modules:
			self.add_module(*stage)  # layer1, layer2, etc
		self.feature_info.extend(stage_feature_info)

		# Head (Pooling and Classifier)
		# self.num_features = 512 * block.expansion
		# self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

		self.init_weights(zero_init_last=zero_init_last)

	@torch.jit.ignore
	def init_weights(self, zero_init_last=True):
		for n, m in self.named_modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
		if zero_init_last:
			for m in self.modules():
				if hasattr(m, 'zero_init_last'):
					m.zero_init_last()

	def reset_classifier(self, num_classes, global_pool='avg'):
		self.num_classes = num_classes

	# self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

	def forward_features(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.act1(x)
		x = self.maxpool(x)

		x1 = self.layer1(x)
		x2 = self.layer2(x1)
		x3 = self.layer3(x2)
		x4 = self.layer4(x3)

		return [x1, x2, x3, x4]

	def forward_head(self, x, pre_logits: bool = False):
		x = self.global_pool(x)
		if self.drop_rate:
			x = F.dropout(x, p=float(self.drop_rate), training=self.training)
		return x if pre_logits else self.fc(x)

	def forward(self, x):
		x = self.forward_features(x)
		x = format_reverse(x)
		# x = self.forward_head(x)
		return x


def resnet_backbone(**kwargs):
	"""Constructs a ResNet-50 model.
	"""
	model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)
	return ResNet(**model_args)


if __name__ == '__main__':
	x = torch.rand(2, 3, 448, 448)
	model = resnet_backbone(num_classes=200)
	y = model(x)
	print(y)
