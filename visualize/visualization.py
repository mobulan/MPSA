import os
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from sklearn.manifold import TSNE
from torchvision import transforms
from models.build import build_models
from setup import config
from models.mps import format_reverse
from utils.data_loader import build_transforms
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, LayerCAM, EigenCAM,EigenGradCAM,GradCAMElementWise
import matplotlib.pyplot as plt


def open_sample_folder(root):
	file_list = os.listdir(root)
	file_list.sort()
	full_path = []
	for file_name in file_list:
		file_path = os.path.join(root, file_name)
		full_path.append(file_path)
	return full_path


def img_transform(file_list):
	train_transforms, test_transforms = build_transforms(config)
	img_list = []
	for f in file_list:
		with open(f, 'r'):
			img = Image.open(f)
		img = test_transforms(img)
		img_list.append(img)
	# plt.imshow(img.permute(1,2,0))
	# plt.show()
	img_list = torch.stack(img_list)
	return img_list


def img_visualize(file_list):
	test_base = [transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC),
	             transforms.RandomRotation((15)),
	             transforms.ColorJitter(0.4,0.4,0.4),
	             transforms.CenterCrop(config.data.img_size)]
	test_transforms = transforms.Compose([*test_base])
	img_list = []
	for f in file_list:
		with open(f, 'r'):
			img = Image.open(f)
		img = test_transforms(img)
		img = np.array(img)
		img_list.append(img)
		plt.imshow(img)
		plt.show()
	img_list = np.stack(img_list)
	return img_list


def build_eval_model(config, num_class=200):
	checkpoint = torch.load(config.model.resume, map_location='cpu')
	state_dicts = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
	try:
		state_dicts = {k.replace('_orig_mod.', ''): v for k, v in state_dicts.items()}
	except:
		pass
	for keys in state_dicts:
		if 'backbone' not in keys:
			pass
	model = build_models(config, num_class)
	model.load_state_dict(state_dicts, strict=False)
	for i in range(4):
		model.block.parts_generation_list[i].assess = True
	model.assess = True
	del checkpoint
	torch.cuda.empty_cache()
	model.eval()
	return model


def class_bar_figure(out, num_classes, softmax=False):
	x = torch.arange(1, num_classes + 1)
	class_name = pd.read_csv('dataset_class_name/classes.txt', header=None, sep=' ')
	class_name = class_name.drop(columns=0).values.squeeze()
	for logits in out:
		index = torch.argmax(logits, -1)
		pred_class = class_name[index]
		print(f'The Prediction Class is: {pred_class}')
		if softmax:
			logits = F.softmax(logits, -1)
			plt.ylim((0, 1))
		plt.figure(figsize=(20, 5))
		plt.bar(x, logits, width=0.9)
		plt.grid(True, alpha=0.5)
		plt.xlim((0, num_classes))
		plt.title(f'{pred_class}')
		plt.annotate(f'{index + 1}', xy=(index + 1, logits[index]))
		plt.tight_layout()
		plt.savefig(f'prediction_result/{pred_class}.pdf')
		plt.show()


def part_generation_pos(model):
	for t in range(4):
		part_pos = model.block.parts_generation_list[t].part_pos
		part_pos = format_reverse(part_pos).squeeze()
		part_pos = part_pos.cpu().detach().numpy()
		os.makedirs(f'part_generation_pos/stage_{t + 1}/', exist_ok=True)
		for i, map in enumerate(part_pos):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(f'part_generation_pos/stage_{t + 1}/map_{i:03}.jpg', map)


def part_attention_pos(model):
	for t in range(4):
		# size [1, self.num_heads, query_size[0] * query_size[1], self.num_parts-self.parts_drop]
		part_pos = model.block.mpsa_list[t].atten_pos
		head_mean = part_pos.mean(1)
		head_mean = format_reverse(head_mean).squeeze()
		head_mean = head_mean.cpu().detach().numpy()
		os.makedirs(f'part_attention_pos/headmean_stage_{t + 1}/', exist_ok=True)
		for i, map in enumerate(head_mean):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(f'part_attention_pos/headmean_stage_{t + 1}/map_{i:03}.jpg', map)
		parts_mean = part_pos.mean(-1).transpose(-2, -1)
		parts_mean = format_reverse(parts_mean).squeeze()
		parts_mean = parts_mean.cpu().detach().numpy()
		os.makedirs(f'part_attention_pos/pratmean_stage_{t + 1}/', exist_ok=True)
		for i, map in enumerate(parts_mean):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(f'part_attention_pos/pratmean_stage_{t + 1}/map_{i:03}.jpg', map)


def sample_map_save(file_list, saved_map):
	imgs = img_visualize(file_list)
	for b in range(imgs.shape[0]):
		for t, stage_map in enumerate(saved_map):
			sample_map = torch.load(stage_map).transpose(-2, -1)
			sample_map = format_reverse(sample_map).cpu().detach().numpy()
			maps = sample_map[b]
			os.makedirs(f'sampling_map/img_{b}/stage_{t + 1}', exist_ok=True)
			for i, map in enumerate(maps):
				map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
				# plt.imshow(f'sampling_map/img_{t}/stage_4/map_{i:03}.jpg', map)
				plt.imshow(imgs[b])
				plt.imshow(map, alpha=0.5,cmap='jet')
				plt.axis('off')
				plt.savefig(f'sampling_map/img_{b}/stage_{t + 1}/map_{i:03}.jpg', bbox_inches='tight', pad_inches=0)
				plt.cla()


def grad_cam_save(file_list,model):
	model = model.eval().cuda()
	# target_layers = [model.backbone.layers[-1].blocks[-1].norm1]
	target_layers = [model.block.activation]
	# target_layers = [model.norm]
	imgs = img_visualize(file_list)
	tensors = img_transform(file_list)
	for b in range(imgs.shape[0]):
		tensor = tensors[b].unsqueeze(0)
		with GradCAM(model=model, target_layers=target_layers, reshape_transform=format_reverse, use_cuda=True) as cam:
			grayscale_cams = cam(input_tensor=tensor, targets=None)
		plt.imshow(imgs[b])
		plt.imshow(grayscale_cams[0], alpha=0.5, cmap='jet')
		plt.axis('off')
		plt.savefig(f'feature_weights/img_{b}/tgcam.jpg', bbox_inches='tight', pad_inches=0)
		plt.cla()


def feature_weights_save(file_list, save_weights, model, gap=True):
	imgs = img_visualize(file_list)
	print(f'Feature Weights coefficient of four layer{model.pooling_weights.squeeze()}')
	weights_path = os.path.join(save_weights, 'weights.pt')
	x_path = os.path.join(save_weights, 'x.pt')
	gcam_path = os.path.join(save_weights, 'gradcam.pt')
	feature_weights, x, gcam = torch.load(weights_path), torch.load(x_path), torch.load(gcam_path)
	for b in range(imgs.shape[0]):
		feature_weights_b = feature_weights.squeeze(-1).permute(1, 0, 2)
		if gap:
			# os.makedirs(f'feature_weights/img_{b}/x/', exist_ok=True)
			# for i in range(1920):
			# 	xx = x[b,i,:]
			# 	fm = xx.reshape(12, 12).cpu().detach().numpy()
			# 	feature_map = cv2.resize(fm, (384, 384))
			# 	plt.imshow(imgs[b])
			# 	plt.imshow(feature_map, alpha=0.5)
			# 	plt.axis('off')
			# 	plt.savefig(f'feature_weights/img_{b}/x/fm{i:04}.jpg', bbox_inches='tight', pad_inches=0)
			# 	plt.cla()
			xm = x.mean(1).squeeze(1)
		else:
			xm, _ = x.max(1)
		fw = feature_weights_b[b].reshape(4, 12, 12).cpu().detach().numpy()
		fm = xm[b].reshape(12, 12).cpu().detach().numpy()
		gc = gcam[b].reshape(12, 12).cpu().detach().numpy()

		os.makedirs(f'feature_weights/img_{b}', exist_ok=True)
		feature_map = cv2.resize(fm, (384, 384), interpolation=cv2.INTER_NEAREST)
		plt.imshow(imgs[b])
		plt.imshow(feature_map, alpha=0.5)
		plt.axis('off')
		plt.savefig(f'feature_weights/img_{b}/feature_map.jpg', bbox_inches='tight', pad_inches=0)
		plt.cla()

		gradcam = cv2.resize(gc, (384, 384))
		plt.imshow(imgs[b])
		plt.imshow(gradcam, alpha=0.5, cmap='jet')
		plt.axis('off')
		plt.savefig(f'feature_weights/img_{b}/gcam.jpg', bbox_inches='tight', pad_inches=0)
		plt.cla()


		for i in range(4):
			plt.imshow(imgs[b])
			nmap = cv2.resize(fw[i], (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imshow(nmap, alpha=0.5)
			plt.axis('off')
			plt.savefig(f'feature_weights/img_{b}/map_{i + 1}.jpg', bbox_inches='tight', pad_inches=0)
			plt.cla()

		plt.imshow(imgs[b])
		sum_fw = (model.pooling_weights.reshape(4,1,1).cpu().detach().numpy() * fw).sum(0)
		tmap = cv2.resize(sum_fw, (384, 384), interpolation=cv2.INTER_NEAREST)
		plt.imshow(tmap, alpha=0.5)
		plt.axis('off')
		plt.savefig(f'feature_weights/img_{b}/sum map.jpg', bbox_inches='tight', pad_inches=0)
		plt.cla()
	pass

def gauss(data):
	data = data.float()
	return torch.exp(-1 * data**2)

def center_norm(datas, center=True):
	if center:
		datas = datas - datas.mean(-2, keepdim=True)
	datas = datas / torch.norm(datas, dim=-1, keepdim=True)
	return datas
def tSNE(dataset,num_class):
	features = torch.load(f'saved_features/{dataset}_f.pth').cpu()
	labels = torch.load(f'saved_features/{dataset}_l.pth').cpu()
	# features = gauss(features)
	# features = center_norm(features)

	onehot = torch.zeros((features.shape[0],num_class))
	onehot.scatter_(1,labels.unsqueeze(1),1)
	print(onehot.shape)
	print(features.shape)
	max_entropy = onehot - F.softmax(features.float(),-1)
	res = F.relu(max_entropy).sum(-1)
	# print(res)

	class_difficult = torch.zeros(num_class)
	for i in range(num_class):
		class_difficult[i] = res[labels == i].mean()
	print(torch.topk(class_difficult, 50, -1))
	# Dog
	# difficlut_class = [99, 113, 80, 97, 28, 29, 114, 81, 16, 47, 98, 115, 45, 36,
	# 35, 50, 53, 117, 8, 37, 15, 4, 34, 84, 7, 89, 11, 49,
	# 90, 42, 17, 32, 76, 20, 19, 21, 68, 71, 38, 72, 14, 112,
	# 57, 51, 91, 95, 48, 1, 105, 3]
	# Car
	# difficlut_class = [63, 70, 21, 18, 41, 74, 13, 82, 30, 61, 15, 83, 44, 16,
	#                    68, 118, 42, 79, 76, 4, 73, 87, 24, 89, 119, 40, 103, 48,
	#                    165, 20, 8, 55, 6, 9, 85, 72, 17, 183, 173, 27, 51, 39,
	#                    58, 32, 53, 12, 35, 28, 180, 75]

	# difficlut_class = [63, 70, 21, 18, 41, 74, 13, 82, 30, 61, 15, 83, 44, 16,
	#  68, 118, 42, 179, 176, 4, 73, 87, 124, 89, 119, 40, 103, 148,
	#  165, 20, 8, 55, 6, 9, 185, 72, 17, 183, 173, 127, 151, 139,
	#  58, 132, 53, 12, 135, 128, 180, 175]
	# CUB
	# difficlut_class = [ 58,  65, 143, 101,  29,  42,  38,  61,   8, 145,  28,  64,  36,  22,
    #      26,  21, 110,  70,  10,  66, 178, 106, 174,  24, 129, 126, 172, 144,
    #      59, 118, 134, 114, 142, 156,  48,  81, 115,  71, 111,  49, 125, 141,
    #      95, 135,  63, 104, 117, 122, 140,  79]
	print(features.shape)
	difficult_class = torch.arange(num_class)
	tsne = TSNE(n_components=2,perplexity=10)  # 降维到2维
	embedded_features = tsne.fit_transform(features)
	colors = plt.cm.rainbow(np.linspace(0, 1,num_class))
	b = torch.randperm(num_class)
	colors = colors[b]
	marker_size = 5
	# Visualize the results
	plt.figure(figsize=(10, 8))
	for i, i_c in enumerate(difficult_class):
		plt.scatter(embedded_features[labels == i_c, 0],
		            embedded_features[labels == i_c, 1],
		            color=colors[i],
		            label=str(i_c), s=marker_size)
		# class_center = np.mean(embedded_features[labels == i], axis=0)
		# plt.scatter(class_center[0], class_center[1], color=colors[i],
		#             marker='*', s=marker_size*10, edgecolor='black', linewidth=0.1)

	plt.legend().set_visible(False)
	plt.axis('off')
	plt.savefig(f'tsne_{dataset}_fff.pdf', bbox_inches='tight', pad_inches=0)
	# plt.legend()
	plt.show()


def model_prediction(file_list, model, label=None):
	model.eval()
	rgb_img = img_visualize(file_list)
	img = img_transform(file_list)
	out = model(img)
	out = out.squeeze().cpu().detach()
	return out


if __name__ == '__main__':
	dataset = 'dogs'
	num_class = {'cub': 200, 'cars': 196, 'dogs': 120}
	file_list = open_sample_folder(f'sample_img/{dataset}')
	model = build_eval_model(config, num_class[dataset])
	label = None
	model.eval()
	out = model_prediction(file_list, model, label)
	print('outlogits done')

	# # 类别预测Logits/概率
	# class_bar_figure(out, num_class[dataset], softmax=False)
	# print('logits figure done')

	# # 部件采样注意力的位置编码
	# part_attention_pos(model)
	# print('part attention pos done')
	#
	# # 部件生成的位置编码
	# part_generation_pos(model)
	# print('part generation pos done')
	#
	# # 根据输入生成的采样图
	# sample_map_save(file_list,open_sample_folder('sampling_map/map_file'))
	# print('sampling map done')
	#
	# 特征权重与特征图
	# feature_weights_save(file_list, 'feature_weights', model, True)
	# print('feature weights done')
	#
	# 梯度类激活图
	# grad_cam_save(file_list,model)
	# print('grad cam done')

	# tSNE(dataset,num_class[dataset])