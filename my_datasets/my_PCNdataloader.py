import torch
from my_PCN import PCN
from utils.misc import *

def get_dataloader(config, args):
    train_dataset = PCN(config.dataset.train._base_, config.dataset.train.others)
    test_dataset = PCN(config.dataset.val._base_, config.dataset.val.others)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.dataset.train.others.bs,
                                                    shuffle = True, 
                                                    drop_last = True,
                                                    num_workers = int(args.num_workers),
                                                    worker_init_fn=worker_init_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                    shuffle = False, 
                                                    drop_last = False,
                                                    num_workers = int(args.num_workers),
                                                    worker_init_fn=worker_init_fn)
            
    return train_dataloader, test_dataloader

