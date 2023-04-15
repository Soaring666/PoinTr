import torch
import os
import wandb
import json
import torch.optim as optim
from utils import parser, misc
from utils.config import *
from tqdm import tqdm
from my_datasets.my_PCN import get_dataloader
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1
from models.my_foldingnet import my_foldingNet


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args, config):

    wandb.login()
    run = wandb.init(project='my_foldingnet')
    start_epoch = 0
    max_epoch = 400
    lr = 1e-4
    weight_decay=1e-6

    model = my_foldingNet().to(device)
    train_dataloader, test_dataloader = get_dataloader(config, args)
    # (train_sampler, train_dataloader), (_, test_dataloader) = dataset_builder(args, config.dataset.train), \
    #                                                         dataset_builder(args, config.dataset.val)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=weight_decay)

    for epoch in range(start_epoch, max_epoch):
        losses = AverageMeter()
        model.train()
        for (taxonomy_ids, model_ids, data) in tqdm(train_dataloader):
            gt = data[1].to(device)
            recon = model(gt)
            loss_func = ChamferDistanceL1().to(device)
            loss = loss_func(recon, gt)
            loss.backward()
            optimizer.step()

            # wandb.log({"train_loss_batch": loss.item()})
            losses.update(loss.item())
        wandb.log({"train_loss_epoch": losses.avg()})

        #test metric
        if epoch % 20 == 0:
            validate(model, test_dataloader, epoch)

        #visulize result
        if epoch % 50 == 0:
            with torch.no_grad():
                (taxonomy_ids, model_ids, data) = next(iter(test_dataloader))
                model.eval()
                gt = data[1].to(device)
                input_pc = gt.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                recon = model(gt)
                recon = recon.squeeze().detach().cpu().numpy()
                recon = misc.get_ptcloud_img(recon)
                wandb.log({"epoch": epoch, "input": wandb.Image(input_pc), "recon": wandb.Image(recon)})


def validate(model, test_dataloader, epoch):
    with torch.no_grad():
        print("Validate the test dataset and visulize points")
        model.eval()
        losses = AverageMeter()
        category_metrics = dict()
        test_metrics = AverageMeter(Metrics.names())

        for (taxonomy_ids, model_ids, data) in tqdm(test_dataloader):
            gt = data[1].to(device)
            recon = model(gt)
            _metrics = Metrics.get(recon, gt)

            #记录每个种类的指标
            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)

        #计算平均指标
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        wandb.log({'test_loss_epoch': test_metrics.avg()[1]})

        #打印结果
        shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
        print('============================ TEST RESULTS ============================')
        print('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]))
        msg = ''
        msg += 'Taxonomy' + '\t'
        msg += 'ClassName' + '\t'
        msg += 'nsamples' + '\t'
        for metric in test_metrics.items:
            msg += metric + '\t'
        print(msg)

        for taxonomy_id in category_metrics:
            msg = ''
            msg += (taxonomy_id + '\t')
            msg += shapenet_dict[taxonomy_id] + '\t'
            msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
            for value in category_metrics[taxonomy_id].avg():
                msg += '%.3f' % value + '\t'
            print(msg)

        msg = ''
        msg += 'Overall\t\t'
        for value in test_metrics.avg():
            msg += '%.3f \t' % value
        print(msg)
        
if __name__ == '__main__':
    args = parser.get_args()
    args.distributed = False
    config = get_config(args)
    config.dataset.train.others.bs = config.total_bs
    main(args, config)


