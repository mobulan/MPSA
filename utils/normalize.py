import torch
import torchvision
from torchvision import transforms
from utils.dataset import *

transform = torchvision.transforms.Compose(
	[transforms.Resize((512, 512)),
	 transforms.ToTensor()]
)


def get_mean_and_std(data, batch_size=8):
	train_loader = torch.utils.data.DataLoader(
		data, batch_size=batch_size, shuffle=False, num_workers=0,
		pin_memory=True)
	mean = torch.zeros(3)
	std = torch.zeros(3)
	p_bar = tqdm(total=len(train_loader), desc='Processing')
	for X, _ in train_loader:
		for d in range(3):
			mean[d] += X[:, d, :, :].mean()
			std[d] += X[:, d, :, :].std()
		p_bar.update(1)
	mean.div_(len(data) / batch_size)
	std.div_(len(data) / batch_size)
	return list(mean.numpy()), list(std.numpy()), len(data)


if __name__ == '__main__':
	root = 'D:\\Experiment\\Datasets\\'
	train = [
		# OxfordIIITPet(root+'pets',True,transform=transform),
		# CUB(root+'CUB_200_2011',True, transform=transform),
		# NABirds(root+'nabirds', True, transform=transform),
		# Dogs(root+'Dogs', True, transform=transform),
		# Cars(root + 'cars', True, transform=transform),
		# Aircraft(root, True, transform=transform),
		# OxfordFlowers(root+'flowers',True,transform),
		Food101(root, True, transform)
	]
	test = [
		# OxfordIIITPet(root+'pets',False,transform=transform),
		# CUB(root+'CUB_200_2011',False, transform=transform),
		# NABirds(root+'nabirds', False, transform=transform),
		# Dogs(root+'Dogs', False, transform=transform),
		# Cars(root + 'cars', False, transform=transform),
		# Aircraft(root, False, transform=transform),
		# OxfordFlowers(root+'flowers',False,transform),
		Food101(root, False, transform)
	]

	for tr, te in zip(train, test):
		mean1, std1, len1 = get_mean_and_std(tr)
		mean2, std2, len2 = get_mean_and_std(te)
		c1, c2 = len1 / (len1 + len2), len2 / (len1 + len2)
		mean = c1 * np.array(mean1) + c2 * np.array(mean2)
		std = c1 * np.array(std1) + c2 * np.array(std2)
		print((*mean, *std))

normalized_info = dict()
normalized_info['pet'] = (0.4817828, 0.4497765, 0.3961324, 0.26035318, 0.25577134, 0.2635264)
normalized_info['cub'] = (0.4865833, 0.5003001, 0.43229204, 0.22157472, 0.21690948, 0.24466534)
normalized_info['nabirds'] = (0.49218804, 0.50868344, 0.46445918, 0.21430683, 0.21335651, 0.25660837)
normalized_info['dogs'] = (0.4764075, 0.45210016, 0.3912831, 0.256719, 0.25130147, 0.25520605)
normalized_info['cars'] = (0.47026777, 0.45981872, 0.4548266, 0.2880712, 0.28685528, 0.29420388)
normalized_info['air'] = (0.47890663, 0.510387, 0.5342661, 0.21548453, 0.2100707, 0.24122715)
normalized_info['flowers'] = (0.4358411, 0.37802523, 0.28822893, 0.29247612, 0.24125645, 0.2639247)

# for v in normalized_info.values():
# 	print(v[:3],v[3:])
