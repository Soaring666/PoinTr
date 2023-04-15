import torch
import os
import wandb
import json
import torch.optim as optim
from PIL import Image
from utils import parser, misc
from utils.config import *
from tools import builder
from tqdm import tqdm
from my_datasets.my_PCN import get_dataloader
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args, config):

    wandb.login()
    run = wandb.init(project='my_foldingnet')
    start_epoch = 0

    model = builder.model_builder(config.model).to(device)
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    optimizer, scheduler = builder.build_opti_sche(model, config)


    for epoch in range(start_epoch, config.max_epoch + 1):
        losses = AverageMeter()
        model.train()
        for (taxonomy_ids, model_ids, data) in tqdm(train_dataloader):
            model.zero_grad()
            gt = data[1].to(device)
            _, recon = model(gt)
            loss_func = ChamferDistanceL1().to(device)
            loss = loss_func(recon, gt)
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
        wandb.log({"train_loss_epoch": losses.avg()})

        #test metric
        if epoch % 20 == 0:
            validate(model, test_dataloader, epoch, args)

        # #visulize result
        # if epoch % 50 == 0:
        #     with torch.no_grad():
        #         (taxonomy_ids, model_ids, data) = next(iter(test_dataloader))
        #         model.eval()
        #         gt = data[1].to(device)
        #         input_pc = gt.squeeze().detach().cpu().numpy()
        #         input_pc = misc.get_ptcloud_img(input_pc)
        #         _, recon = model(gt)
        #         recon = recon.squeeze().detach().cpu().numpy()
        #         recon = misc.get_ptcloud_img(recon)
        #         wandb.log({"epoch": epoch, "input": wandb.Image(input_pc), "recon": wandb.Image(recon)})


def validate(model, test_dataloader, epoch, args):
    with torch.no_grad():
        print("Validate the test dataset and visulize points")
        model.eval()
        losses = AverageMeter()
        category_metrics = dict()
        test_metrics = AverageMeter(Metrics.names())

        #创建保存图片的文件夹
        save_pth = os.path.join(args.saveimg_path, 'epoch%d' % epoch)
        os.makedirs(save_pth, exist_ok=True)
        gt_pth = os.path.join(save_pth, 'gt')
        os.makedirs(gt_pth, exist_ok=True)
        recon_pth = os.path.join(save_pth, 'recon')
        os.makedirs(recon_pth, exist_ok=True)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            gt = data[1].to(device)
            _, recon = model(gt)
            _metrics = Metrics.get(recon, gt)

            #记录每个种类的指标
            for _taxonomy_id in taxonomy_ids:
                if _taxonomy_id not in category_metrics:
                    category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[_taxonomy_id].update(_metrics)
            
            #保存图片
            if idx % 500 == 0:
                input_gt = gt.squeeze().detach().cpu().numpy()
                input_gt = misc.get_ptcloud_img(input_gt)
                im = Image.fromarray(input_gt)
                gt_savepth = os.path.join(gt_pth, '%s.jpg' % model_ids)
                im.save(gt_savepth)

                recon = recon.squeeze().detach().cpu().numpy()
                recon = misc.get_ptcloud_img(recon)
                im = Image.fromarray(recon)
                recon_savepth = os.path.join(recon_pth, '%s.jpg' % model_ids)
                im.save(recon_savepth)

                

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

        #使用wandb track
        columns = ['Taxonomy', 'ClassName', 'nsamples', 'F1_Score', 'CDL1', 'CDL2', 'EMD']
        data = []
        for taxonomy_id in category_metrics:
            data_id = [taxonomy_id, shapenet_dict[taxonomy_id], str(category_metrics[taxonomy_id].count(0))]
            for value in category_metrics[taxonomy_id].avg():
                data_id.append(value)
            data.append(data_id)
        data_all = ['overall', '', '']
        for value in test_metrics.avg():
            data_all.append(value)
        data.append(data_all)

        table = wandb.Table(data=data, columns=columns)
        wandb.log({'test_table': table})
        

        
if __name__ == '__main__':
    args = parser.get_args()
    args.distributed = False
    config = get_config(args)
    config.dataset.train.others.bs = config.total_bs
    main(args, config)


