# import sys
# sys.path.append('..')
# from baseline import baseline_models
import os
import timm
import torch
from models.backbone.ResNet import resnet_backbone
from models.backbone.Swin_Transformer import swin_backbone, swin_backbone_large,swin_backbone_tiny
from models.backbone.Vision_Transformer import vit_backbone
from models.mps import MultiPartsSampling


def build_models(config, num_classes):
	if config.model.baseline_model:
		model = baseline_models(config, num_classes)
		load_pretrained(config, model)
		return model
	dim, backbone, backbone_type = 512, None, 'hier'
	if config.model.type.lower() == 'resnet':
		dim = 2048
		backbone = resnet_backbone(num_classes=num_classes, drop_path_rate=config.model.drop_path)

	elif config.model.type.lower() == 'swin':
		if config.model.name.lower() == 'swin tiny':
			dim = 768
			backbone = swin_backbone_tiny(num_classes=num_classes, drop_path_rate=config.model.drop_path,
			                              img_size=config.data.img_size,window_size=config.data.img_size // 32)
		elif config.model.name.lower() == 'swin large':
			dim = 1536
			backbone = swin_backbone_large(num_classes=num_classes, drop_path_rate=config.model.drop_path,
			                              img_size=config.data.img_size, window_size=config.data.img_size // 32)
		else:
			dim = 1024
			backbone = swin_backbone(num_classes=num_classes, drop_path_rate=config.model.drop_path,
			                         img_size=config.data.img_size, window_size=config.data.img_size // 32)

	elif config.model.type.lower() == 'vit':
		dim = 768
		backbone_type = 'vit'
		backbone = vit_backbone(num_classes=num_classes)

	elif config.model.type.lower() == 'swinv2':
		dim = 1536

		backbone = swin_backbone_large(num_classes=num_classes, drop_path_rate=config.model.drop_path,
		                               img_size=config.data.img_size, window_size=config.data.img_size // 32,
		                               cross_layer=config.parameters.cross_layer)

	elif config.model.type.lower() == 'deit':
		dim = 1536
		backbone_type = 'vit'
		backbone = swin_backbone_large(num_classes=num_classes, drop_path_rate=config.model.drop_path,
		                               img_size=config.data.img_size, window_size=config.data.img_size // 32,
		                               cross_layer=config.parameters.cross_layer)
	# try:
	load_pretrained(config, backbone)
	# except:
	# 	print('=' * 20, f'No Pretrained Model has been loaded!'.center(38), '=' * 20)
	model = MultiPartsSampling(dim, config.data.img_size,
	                           backbone, config.parameters.parts_ratio, config.parameters.num_heads,
	                           config.parameters.fwp,config.parameters.att_drop,
	                           config.parameters.head_drop,config.parameters.parts_drop, num_classes,
	                           config.parameters.pos, config.parameters.parts_base, config.parameters.cross_layer,
	                           config.model.label_smooth,mixup=config.data.mixup,backbone_type=backbone_type)
	return model


def baseline_models(config, num_classes):
	model = None
	type = config.model.type.lower()
	if type == 'resnet':
		model = timm.models.create_model('resnet50', pretrained=False, num_classes=num_classes)

	elif type == 'vit':
		model = timm.models.create_model('vit_base_patch16_224_in21k', pretrained=False,
		                                 num_classes=num_classes, img_size=config.data.img_size)
	elif type == 'swin':
		if config.model.name.lower() == 'swin tiny':
			# model = timm.models.create_model('swin_tiny_patch4_window7_224', pretrained=False,
			#                                  num_classes=num_classes, drop_path_rate=config.model.drop_path,
			#                                  img_size=config.data.img_size,window_size=config.data.img_size // 32)
			model = swin_backbone_tiny(num_classes=num_classes, drop_path_rate=config.model.drop_path,
			                           img_size=config.data.img_size,window_size=config.data.img_size // 32,
			                           cross_layer=False)
		else:
			model = timm.models.create_model('swin_base_patch4_window12_384_in22k', pretrained=False,
			                                 num_classes=num_classes, drop_path_rate=config.model.drop_path)
	elif type == 'maxvit':
		model = timm.models.create_model('maxvit_tiny_rw_224', pretrained=False, num_classes=200, img_size=384,
		                                 drop_path_rate=config.model.drop_path)
	elif type == 'swinv2':
		model = timm.models.create_model('swin_large_patch4_window12_384_in22k', pretrained=False,
		                                 num_classes=num_classes, drop_path_rate=config.model.drop_path)
	# print(model)
	return model


def load_pretrained(config, model):
	if config.local_rank in [-1, 0]:
		print('-' * 11, f'Loading weight \'{config.model.pretrained}\' for fine-tuning'.center(56), '-' * 11)

	if os.path.splitext(config.model.pretrained)[-1].lower() in ('.npz', '.npy'):
		# numpy checkpoint, try to load via model specific load_pretrained fn
		if hasattr(model, 'load_pretrained'):
			model.load_pretrained(config.model.pretrained)
			if config.local_rank in [-1, 0]:
				print('-' * 18, f'Loaded successfully \'{config.model.pretrained}\''.center(42), '-' * 18)

			torch.cuda.empty_cache()
			return

	checkpoint = torch.load(config.model.pretrained, map_location='cpu')
	state_dict = None
	type = config.model.type.lower()

	if type == 'maxvit':
		state_dict = checkpoint
		del state_dict['head.fc.weight']
		del state_dict['head.fc.bias']
		if config.model.baseline_model:
			torch.nn.init.constant_(model.head.fc.bias, 0.)
			torch.nn.init.constant_(model.head.fc.weight, 0.)
		relative_position_index_keys = [k for k in state_dict.keys() if "rel_pos" in k]
		for k in relative_position_index_keys:
			del state_dict[k]

	if type == 'resnet':
		try:
			state_dict = checkpoint['state_dict']
		except:
			state_dict = checkpoint
		# print(state_dict.keys())
		del state_dict['fc.weight']
		del state_dict['fc.bias']
		if config.model.baseline_model:
			torch.nn.init.constant_(model.fc.bias, 0.)
			torch.nn.init.constant_(model.fc.weight, 0.)

	# fc_pretrained = state_dict['fc.bias']
	# Nc1 = fc_pretrained.shape[0]
	# Nc2 = model.fc.bias.shape[0]
	# if Nc1!=Nc2:
	# 	torch.nn.init.constant_(model.fc.bias, 0.)
	# 	torch.nn.init.constant_(model.fc.weight, 0.)
	# 	del state_dict['fc.weight']
	# 	del state_dict['fc.bias']

	elif type == 'swin' or type == 'swinv2':
		state_dict = checkpoint['model']
		# delete relative_position_index since we always re-init it
		relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
		for k in relative_position_index_keys:
			del state_dict[k]

		# delete relative_coords_table since we always re-init it
		relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
		for k in relative_position_index_keys:
			del state_dict[k]

		# delete attn_mask since we always re-init it
		attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
		for k in attn_mask_keys:
			del state_dict[k]

		# Modify Patch_Merging
		if not config.model.baseline_model or config.model.name.lower() == 'swin tiny':
			patch_merging_keys = [k for k in state_dict.keys() if "downsample" in k]
			patch_merging_pretrained = []
			new_keys = []
			for k in patch_merging_keys:
				patch_merging_pretrained.append(state_dict[k])
				del state_dict[k]
				k = k.replace(k[7], f'{int(k[7]) + 1}')
				new_keys.append(k)

			for nk, nv in zip(new_keys, patch_merging_pretrained):
				state_dict[nk] = nv
			# print(patch_merging)

		# bicubic interpolate relative_position_bias_table if not match
		relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
		# relative_position_bias_table_keys = [x for x in relative_position_bias_table_keys if 'layers.3.' not in x]
		for k in relative_position_bias_table_keys:
			relative_position_bias_table_pretrained = state_dict[k]
			relative_position_bias_table_current = model.state_dict()[k]
			L1, nH1 = relative_position_bias_table_pretrained.size()
			L2, nH2 = relative_position_bias_table_current.size()

			if nH1 != nH2:
				print(f"Error in loading {k}, passing......")
			else:
				if L1 != L2:
					# bicubic interpolate relative_position_bias_table if not match
					S1 = int(L1 ** 0.5)
					S2 = int(L2 ** 0.5)
					relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
						relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
						mode='bicubic')

					state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
		# bicubic interpolate absolute_pos_embed if not match
		absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
		for k in absolute_pos_embed_keys:
			# dpe
			absolute_pos_embed_pretrained = state_dict[k]
			absolute_pos_embed_current = model.state_dict()[k]
			_, L1, C1 = absolute_pos_embed_pretrained.size()
			_, L2, C2 = absolute_pos_embed_current.size()
			if C1 != C1:
				print(f"Error in loading {k}, passing......")
			else:
				if L1 != L2:
					S1 = int(L1 ** 0.5)
					S2 = int(L2 ** 0.5)
					absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
					absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
					absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
						absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
					absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
					absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
					state_dict[k] = absolute_pos_embed_pretrained_resized

		# # check classifier, if not match, then re-init classifier to zero
		# head_bias_pretrained = state_dict['head.bias']
		# Nc1 = head_bias_pretrained.shape[0]
		# Nc2 = model.head.bias.shape[0]
		# if (Nc1 != Nc2):
		if config.model.baseline_model:
			torch.nn.init.constant_(model.head.bias, 0.)
			torch.nn.init.constant_(model.head.weight, 0.)
		del state_dict['head.weight']
		del state_dict['head.bias']

	msg = model.load_state_dict(state_dict, strict=False)
	# print(msg)
	if config.local_rank in [-1, 0]:
		print('-' * 16, ' Loaded successfully \'{:^22}\' '.format(config.model.pretrained), '-' * 16)

	del checkpoint
	torch.cuda.empty_cache()


def freeze_backbone(model, freeze_params=False):
	if freeze_params:
		for name, parameter in model.named_parameters():
			if name.startswith('backbone'):
				parameter.requires_grad = False


if __name__ == '__main__':
	model = build_models(1, 200)
	print(model)
