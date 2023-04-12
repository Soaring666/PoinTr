import torch
import os
from tqdm import tqdm
from models.my_foldingnet import my_foldingNet
from utils import parser, misc
from utils.config import *
# from datasets import data_transforms
import transforms3d
from my_datasets.my_PCN import PCN
from utils.misc import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = parser.get_args()
args.distributed = False
config = get_config(args)
config.dataset.train.others.bs = config.total_bs
model = my_foldingNet().to(device)



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
#     pass

for i in range(4):
    a = torch.randn(4, 2048, 3).to(device)
    b = torch.randn(4, 2048, 3).to(device)
    d = model(a)
    loss = model.get_loss(a, b)
    print(loss.device)
