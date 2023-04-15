import torch
import os
import numpy as np
import wandb
from tqdm import tqdm
from utils import parser, dist_utils, misc
from utils.config import *
from my_datasets.my_PCN import PCN
from utils.misc import *
from tools import builder
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.get_args()
args.distributed = False
config = get_config(args)
config.dataset.train.others.bs = config.total_bs
model = builder.model_builder(config.model).to(device)
(train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                        builder.dataset_builder(args, config.dataset.val)

a = torch.randn(100, 3)
a = a.numpy()
x, y, z = a.transpose(1, 0)
input_gt = misc.get_ptcloud_img(a)
im = Image.fromarray(input_gt)
im.show()
print(type(im))


# image = wandb.Image(input_gt)
# run = wandb.init(project='image')
# wandb.log({'image': image})

# for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(train_dataloader)):
#     if idx % 500 == 0:
#         gt = data[1].to(device)
#         _, recon = model(gt)

#         input_gt = gt.squeeze().detach().cpu().numpy()
#         input_gt = misc.get_ptcloud_img(input_gt)
#         val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')
#     pass



# train_dataset = PCN(config.dataset.train._base_, config.dataset.train.others)
# test_dataset = PCN(config.dataset.val._base_, config.dataset.val.others)
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.dataset.train.others.bs,
#                                                 shuffle = True, 
#                                                 drop_last = True,
#                                                 num_workers = int(args.num_workers),
#                                                 worker_init_fn=worker_init_fn)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
#                                                 shuffle = False, 
#                                                 drop_last = False,
#                                                 num_workers = int(args.num_workers),
#                                                 worker_init_fn=worker_init_fn)

# for date in tqdm(train_dataloader):
# #     pass
# epoch = 3
# metrics = [0.111, 0.222, 0.333, 0.000]
# test_metrics = ['F1_score', 'CDL1', 'CDL2', 'EMDinstance']
# print('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in metrics]))
# msg = ''
# msg += 'Taxonomy' + '\t'
# msg += 'ClassName' + '\t'
# msg += 'nsamples' + '\t'
# for metric in test_metrics:
#     msg += metric + '\t'
# print(msg)


# for i in range(4):
#     a = torch.randn(4, 2048, 3).to(device)
#     b = torch.randn(4, 2048, 3).to(device)
#     d = model(a)
#     loss = model.get_loss(a, b)
#     print(loss.device)
