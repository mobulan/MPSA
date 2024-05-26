import random
import socket

import numpy as np
import yaml

from utils.eval import get_world_size
from utils.info import *
from settings.defaults import augment_parser


def SetupConfig(config, cfg_file=None, args=None):
	def _check_args(name):
		if hasattr(args, name) and f'args.{name}' is not None:
			return True
		return False

	args = augment_parser()
	if cfg_file:
		config.defrost()
		print('-' * 18, f'Merge From {cfg_file}'.center(42), '-' * 18)
		config.merge_from_file(cfg_file)
		config.freeze()
	if args:
		print('-' * 18, 'Merge From Argument parser'.center(42), '-' * 18)
		config.defrost()
		if _check_args('parts_drop') and args.parts_drop is not None:
			config.parameters.parts_drop = args.parts_drop
		if _check_args('parts_ratio') and args.parts_ratio is not None:
			config.parameters.parts_ratio = args.parts_ratio
		config.freeze()
	return config


def SetupLogs(config, rank=0):
	write = config.write
	if rank not in [-1, 0]: return
	# 建立log对象
	if write:
		os.makedirs(config.data.log_path, exist_ok=True)
	log = Log(fname=config.data.log_path, write=write)
	# 输出
	# PTitle(log,config.local_rank)
	PSetting(log, 'Data Settings', config.data.keys(), config.data.values(), newline=2, rank=config.local_rank)
	PSetting(log, 'Hyper Parameters', config.parameters.keys(), config.parameters.values(), rank=config.local_rank)
	PSetting(log, 'Training Settings', config.train.keys(), config.train.values(), rank=config.local_rank)
	PSetting(log, 'Other Settings', config.misc.keys(), config.misc.values(), rank=config.local_rank)

	# 返回log实例
	return log


def SetupDevice():
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ['WORLD_SIZE'])
		torch.cuda.set_device(rank)
		torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
		torch.distributed.barrier()
	else:
		rank = -1
		world_size = -1
	nprocess = torch.cuda.device_count()
	torch.cuda.set_device(rank)
	# torch.use_deterministic_algorithms(True)
	# torch.backends.cudnn.benchmark = True
	return nprocess, rank


def SetSeed(config):
	seed = config.misc.seed + config.local_rank
	# torch.manual_seed(seed)
	# torch.cuda.manual_seed(seed)
	# np.random.seed(seed)
	# random.seed(seed)


def ScaleLr(config):
	base_lr = config.train.lr * config.data.batch_size * get_world_size() / 512.0
	return base_lr


def LocateDatasets(config=None):
	def HostIp():
		"""
		查询本机ip地址
		:return: ip
		"""
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(('8.8.8.8', 80))
			ip = s.getsockname()[0]
		finally:
			s.close()
		return ip

	ip = HostIp()
	print(ip)
	address = ip.split('.')[3]
	data_root = config.data.data_root
	batch_size = config.data.batch_size
	if ip == '210.45.215.179':
		data_root = '/DATA/meiyiming/ly/dataset'
		batch_size = config.data.batch_size // 2
	elif ip == '172.17.71.118':
		data_root = '/home/cvpr/dataset/'
	# Add Your IP and specific the path of dataset If you have multiple servers
	
	# print(data_root,batch_size)
	return data_root, batch_size


if __name__ == '__main__':
	LocateDatasets()
