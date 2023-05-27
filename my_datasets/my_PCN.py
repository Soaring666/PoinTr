import torch.utils.data as data
import json
from datasets import data_transforms
import random
import torch
import numpy as np
from utils.misc import *
from datasets.io import IO
from utils.logger import *
from utils.config import merge_new_config


class PCN(data.Dataset):
    # def __init__(self, data_root, subset, class_choice = None):
    def __init__(self, config, add_config=None):
        if add_config is not None:
            config = merge_new_config(config, add_config)

        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.cars = config.CARS

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

        ######get embedding
        self.catfile = config.CATFILE
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        
    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:  #dc为字典
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']), logger='PCNDATASET')
            samples = dc[subset]

            for s in samples:
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'taxonomy_name':
                    dc['taxonomy_name'],
                    'model_id':
                    s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path':
                    self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0

        for ri in ['partial', 'gt']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            data[ri] = IO.get(file_path).astype(np.float32)

        assert data['gt'].shape[0] == self.npoints

        if self.transforms is not None:
            data = self.transforms(data)

        return self.classes[sample['taxonomy_name']], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)


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

