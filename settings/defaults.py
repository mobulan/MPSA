from yacs.config import CfgNode as CN
import argparse

_C = CN()
# -----------------------------------------------------------------------------
# Data Settings
# -----------------------------------------------------------------------------
_C.data = CN()
_C.data.dataset = 'cub'
_C.data.batch_size = 8
_C.data.data_root = '/data/datasets/fine-grained'
_C.data.img_size = 384
_C.data.resize = int(_C.data.img_size / 0.75)
_C.data.padding = 0
_C.data.no_crop = False
_C.data.autoaug = False
_C.data.blur = 0.  # 0.1
_C.data.color = 0.  # 0.2
_C.data.hue = 0.  # 0.4
_C.data.rotate = 0
_C.data.mixup = 0.  # 0.8
_C.data.cutmix = 0.  # 1.0

# -----------------------------------------------------------------------------
# Model Settings
# -----------------------------------------------------------------------------
_C.model = CN()
_C.model.type = 'ResNet'
_C.model.name = 'ResNet-50'
_C.model.baseline_model = True
_C.model.pretrained = 'pretrained'
_C.model.pre_version = ''
_C.model.pre_suffix = '.pth'
# _C.model.mps_pretrained = 'pretrained/Swin Base.pth'
_C.model.resume = ''
_C.model.num_classes = 1000
_C.model.drop_path = 0.1
_C.model.dropout = 0.0
_C.model.label_smooth = 0.0
_C.model.parameters = 0

# -----------------------------------------------------------------------------
# Parameters Settings
# -----------------------------------------------------------------------------
_C.parameters = CN()
_C.parameters.parts_ratio = 4
_C.parameters.num_heads = 16
_C.parameters.att_drop = 0.
_C.parameters.head_drop = 0.
_C.parameters.parts_drop = 0.
_C.parameters.fwp = True
_C.parameters.pos = True
_C.parameters.parts_base = 0.
_C.parameters.cross_layer = False
# -----------------------------------------------------------------------------
# Training Settings
# -----------------------------------------------------------------------------
_C.train = CN()
_C.train.start_epoch = 0
_C.train.epochs = 50
_C.train.warmup_epochs = 0
_C.train.weight_decay = 1e-08
_C.train.clip_grad = None
_C.train.checkpoint = False
_C.train.lr = 2e-02
_C.train.scheduler = 'cosine'
_C.train.optimizer = 'SGD'
_C.train.freeze_backbone = False
_C.train.eps = 1e-8
_C.train.betas = (0.9, 0.999)
_C.train.momentum = 0.9

# -----------------------------------------------------------------------------
# Misc Settings
# -----------------------------------------------------------------------------
_C.misc = CN()
_C.misc.amp = True
_C.misc.output = './output'
_C.misc.exp_name = _C.data.dataset
_C.misc.log_name = 'base'
_C.data.log_path = ''
_C.misc.eval_every = 1
_C.misc.seed = 42
_C.misc.eval_mode = False
_C.misc.throughput = False
_C.misc.fused_window = True

_C.write = False
_C.local_rank = -1
_C.device = 'cuda'
_C.cuda_visible = '0,1'


# -----------------------------------------------------------------------------
def augment_parser():
	parser = argparse.ArgumentParser(description='Batch Experiments with Different Parameters')
	parser.add_argument('--parts_ratio', type=int, help='Part sampling ratio r.')
	parser.add_argument('--parts_drop', type=float, help='Part Dropout rate.')
	parser.add_argument('--local_rank', default=-1, type=float, help='1.8.1 old problem')
	args = parser.parse_args()
	return args
