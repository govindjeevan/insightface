import numpy as np
import os
from easydict import EasyDict as edict

config = edict()

config.bn_mom = 0.9
config.workspace = 256
config.emb_size = 512
config.ckpt_embedding = False
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_blocks = [1,4,6,2]
config.net_output = 'E'
config.net_multiplier = 1.0
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.ce_loss = True
config.fc7_lr_mult = 1.0
config.fc7_wd_mult = 1.0
config.fc7_no_bias = False
config.max_steps = 0
config.data_rand_mirror = True
config.data_cutoff = False
config.data_color = 0
config.data_images_filter = 0
config.count_flops = True
config.memonger = False #not work now


# network settings
network = edict()

network.r100 = edict()
network.r100.net_name = 'fresnet'
network.r100.num_layers = 100

network.r100fc = edict()
network.r100fc.net_name = 'fresnet'
network.r100fc.num_layers = 100
network.r100fc.net_output = 'FC'

network.r50 = edict()
network.r50.net_name = 'fresnet'
network.r50.num_layers = 50

network.r50v1 = edict()
network.r50v1.net_name = 'fresnet'
network.r50v1.num_layers = 50
network.r50v1.net_unit = 1

network.a56 = edict()
network.a56.net_name = 'fresattnet'
network.a56.num_layers = 56

network.a92 = edict()
network.a92.net_name = 'fresattnet'
network.a92.num_layers = 92


network.da92 = edict()
network.da92.net_name = 'fdensenetatt'
network.da92.num_layers = 92

network.cbamd74 = edict()
network.cbamd74.net_name = 'fcbamdense'
network.cbamd74.num_layers = 74

network.cbamd152 = edict()
network.cbamd152.net_name = 'fcbamdense'
network.cbamd152.num_layers = 152

network.da90 = edict()
network.da90.net_name = 'fcbam'
network.da90.num_layers = 90

network.ra200 = edict()
network.ra200.net_name = 'fcbam'
network.ra200.num_layers = 200

network.ra152 = edict()
network.ra152.net_name = 'fcbam'
network.ra152.num_layers = 152

network.d169 = edict()
network.d169.net_name = 'fdensenet'
network.d169.num_layers = 169
network.d169.per_batch_size = 64
network.d169.densenet_dropout = 0.0

network.d201 = edict()
network.d201.net_name = 'fdensenet'
network.d201.num_layers = 201
network.d201.per_batch_size = 32
network.d201.densenet_dropout = 0.0

network.y1 = edict()
network.y1.net_name = 'fmobilefacenet'
network.y1.emb_size = 128
network.y1.net_output = 'GDC'

network.y2 = edict()
network.y2.net_name = 'fmobilefacenet'
network.y2.emb_size = 256
network.y2.net_output = 'GDC'
network.y2.net_blocks = [2,8,16,4]

network.m1 = edict()
network.m1.net_name = 'fmobilenet'
network.m1.emb_size = 256
network.m1.net_output = 'GDC'
network.m1.net_multiplier = 1.0

network.m05 = edict()
network.m05.net_name = 'fmobilenet'
network.m05.emb_size = 256
network.m05.net_output = 'GDC'
network.m05.net_multiplier = 0.5

network.mnas = edict()
network.mnas.net_name = 'fmnasnet'
network.mnas.emb_size = 256
network.mnas.net_output = 'GDC'
network.mnas.net_multiplier = 1.0

network.mnas05 = edict()
network.mnas05.net_name = 'fmnasnet'
network.mnas05.emb_size = 256
network.mnas05.net_output = 'GDC'
network.mnas05.net_multiplier = 0.5

network.mnas025 = edict()
network.mnas025.net_name = 'fmnasnet'
network.mnas025.emb_size = 256
network.mnas025.net_output = 'GDC'
network.mnas025.net_multiplier = 0.25

network.vargfacenet = edict()
network.vargfacenet.net_name = 'vargfacenet'
network.vargfacenet.net_multiplier = 1.25
network.vargfacenet.emb_size = 512
network.vargfacenet.net_output='J'

# dataset settings
dataset = edict()

dataset.emore = edict()
dataset.emore.dataset = 'emore'
dataset.emore.dataset_path = '../datasets/faces_emore'
dataset.emore.num_classes = 85742
dataset.emore.image_shape = (112,112,3)
dataset.emore.val_targets = ['lfw-custom','lfw']

dataset.casia = edict()
dataset.casia.dataset = 'casia'
dataset.casia.dataset_path = '../datasets/faces_webface_112x112'
dataset.casia.num_classes = 10575
dataset.casia.image_shape = (112,112,3)
dataset.casia.val_targets = ['lfw-new-custom', 'lfw']

dataset.casia_retina = edict()
dataset.casia_retina.dataset = 'casia_retina'
dataset.casia_retina.dataset_path = '/storage/MysterioNet/datasets/maskless/casia-retina'
dataset.casia_retina.num_classes = 10575
dataset.casia_retina.image_shape = (112,112,3)
dataset.casia_retina.val_targets = ['lfw-new-custom', 'lfw']

dataset.casia_retina_masked = edict()
dataset.casia_retina_masked.dataset = 'casia_retina_masked'
dataset.casia_retina_masked.dataset_path = '/storage/MysterioNet/datasets/smdf/original/casia-retina'
dataset.casia_retina_masked.num_classes = 10575
dataset.casia_retina_masked.image_shape = (112,112,3)
dataset.casia_retina_masked.val_targets = ['lfw-masked','lfw']


dataset.retina = edict()
dataset.retina.dataset = 'retina'
dataset.retina.dataset_path = '../datasets/ms1m-retinaface-t1'
dataset.retina.num_classes = 93431
dataset.retina.image_shape = (112,112,3)
dataset.retina.val_targets = ['lfw', 'cfp_fp', 'agedb_30']

loss = edict()
loss.softmax = edict()
loss.softmax.loss_name = 'softmax'

loss.nsoftmax = edict()
loss.nsoftmax.loss_name = 'margin_softmax'
loss.nsoftmax.loss_s = 64.0
loss.nsoftmax.loss_m1 = 1.0
loss.nsoftmax.loss_m2 = 0.0
loss.nsoftmax.loss_m3 = 0.0

loss.arcface = edict()
loss.arcface.loss_name = 'margin_softmax'
loss.arcface.loss_s = 64.0
loss.arcface.loss_m1 = 1.0
loss.arcface.loss_m2 = 0.5
loss.arcface.loss_m3 = 0.0

loss.cosface = edict()
loss.cosface.loss_name = 'margin_softmax'
loss.cosface.loss_s = 64.0
loss.cosface.loss_m1 = 1.0
loss.cosface.loss_m2 = 0.0
loss.cosface.loss_m3 = 0.35

loss.combined = edict()
loss.combined.loss_name = 'margin_softmax'
loss.combined.loss_s = 64.0
loss.combined.loss_m1 = 1.0
loss.combined.loss_m2 = 0.3
loss.combined.loss_m3 = 0.2

loss.triplet = edict()
loss.triplet.loss_name = 'triplet'
loss.triplet.images_per_identity = 5
loss.triplet.triplet_alpha = 0.3
loss.triplet.triplet_bag_size = 7200
loss.triplet.triplet_max_ap = 0.0
loss.triplet.per_batch_size = 33
loss.triplet.lr = 0.05

loss.atriplet = edict()
loss.atriplet.loss_name = 'atriplet'
loss.atriplet.images_per_identity = 5
loss.atriplet.triplet_alpha = 0.35
loss.atriplet.triplet_bag_size = 7200
loss.atriplet.triplet_max_ap = 0.0
loss.atriplet.per_batch_size = 60
loss.atriplet.lr = 0.05

# default settings
default = edict()

# default network
default.network = 'r100'
default.pretrained = ''
default.pretrained_epoch = 1
# default dataset
default.dataset = 'emore'
default.loss = 'arcface'
default.frequent = 20
default.verbose = 1000
default.kvstore = 'device'

default.end_epoch = 10000
default.lr = 0.001
default.wd = 0.0005
default.mom = 0.9
default.per_batch_size = 128
default.ckpt = 3
default.lr_steps = '100000,160000,220000'
default.models_root = './models'


def generate_config(_network, _dataset, _loss):
    for k, v in loss[_loss].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in network[_network].items():
      config[k] = v
      if k in default:
        default[k] = v
    for k, v in dataset[_dataset].items():
      config[k] = v
      if k in default:
        default[k] = v
    config.loss = _loss
    config.network = _network
    config.dataset = _dataset
    config.num_workers = 1
    if 'DMLC_NUM_WORKER' in os.environ:
      config.num_workers = int(os.environ['DMLC_NUM_WORKER'])

