import torch
import torch.nn as nn
import os
import wandb
import json
import time
import copy
from tqdm import tqdm
from PIL import Image
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from warmup_scheduler import GradualWarmupScheduler
from models.seed_utils.utils import fps_subsample

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def run_net(args, config, train_writer=None, val_writer=None):

    if args.local_rank == 0:
        run = wandb.init(project='diffusion')

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
                                                    
    # build model
    premodel = builder.model_builder(config.premodel)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        premodel = premodel.cuda()
        base_model = base_model.cuda()

    #load pretrained model
    builder.load_model(premodel, config.premodel_ckpts, logger=logger) 
    for param in premodel.parameters():
        param.requires_grad = False
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    if config.GradualWarmupScheduler is not None:
        warmup_scheduler = config.GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_scheduler.multiplier, 
                                           total_epoch=warmup_scheduler.total_epoch,
                                           after_scheduler=scheduler)
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = AverageMeter()

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(train_dataloader)):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if  'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShapeNet' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    # partial, gt = misc.random_scale(partial, gt) # specially for KITTI finetune
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            torch.cuda.empty_cache()

            num_iter += 1
            seed, seed_feat, pred_pcds = premodel.forward_encoder(partial)           
            gt_2 = fps_subsample(gt, 2048)
            gt_1 = fps_subsample(gt_2, 512)
            B, N, C = gt_1.shape
            dif_shape = [config.dataset.val.others.bs, N, C]

            mseloss = base_model(gt_1, seed_feat)
            
            train_loss.update(mseloss)
            mseloss.backward()

            if args.distributed:
                dist.barrier()


           # forward
            if num_iter == config.step_per_update:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            
            if args.distributed:
                torch.cuda.synchronize()

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %.4f lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            mseloss.item(), optimizer.param_groups[0]['lr']), logger = logger)

            break

        if args.local_rank == 0:
            wandb.log({"train_mse_loss": train_loss.avg()})


        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) mse_loss = %.4f' %
            (epoch,  epoch_end_time - epoch_start_time, train_loss.avg()), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(premodel, dif_shape, base_model, test_dataloader, epoch, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 2:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(premodel, shape, base_model, test_dataloader, epoch, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_loss_list = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_loss_sum = AverageMeter()

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # 1200, bs is 1

    interval =  n_samples // 10

    #创建保存图片的文件夹
    save_pth = os.path.join(args.saveimg_path, 'epoch%d' % epoch)
    os.makedirs(save_pth, exist_ok=True)
    gt_pth = os.path.join(save_pth, 'gt')
    os.makedirs(gt_pth, exist_ok=True)
    recon_pth = os.path.join(save_pth, 'recon')
    os.makedirs(recon_pth, exist_ok=True)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            # taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            # model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if 'PCN' in dataset_name or dataset_name == 'Completion3D' or 'ProjectShapeNet' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            torch.cuda.empty_cache()


            seed, seed_feat, pred_pcds = premodel.forward_encoder(partial)           
            x_i, x_i_store_list = base_model.module.sample(seed_feat, shape, guide_w = 2.0)
            x_i = x_i.transpose(-2, -1)
            pred_pcds = premodel.forward_decoder(gt=None, gt_pre=x_i, seed=seed,
                                                 seed_feat=seed_feat, pred_pcds=pred_pcds)

            dense_points = pred_pcds[-1]
            recon = copy.deepcopy(dense_points)

            loss_sum, loss_list, gt_fps_list = premodel.get_loss(pred_pcds, partial, gt)
            test_loss_sum.update(loss_sum)
            test_loss_list.update(loss_list)
            
            #保存图片
            if idx % 10 == 0:

                test_gt_list = [gt_fps_list[i][0].squeeze().detach().cpu().numpy() for i in range(4)]
                test_recon_list = [pred_pcds[i][0].squeeze().detach().cpu().numpy() for i in range(4)]  
                if args.local_rank == 0:
                    wandb.log({'test_gt': [wandb.Object3D(i) for i in test_gt_list],
                        'test_recon': [wandb.Object3D(i) for i in test_recon_list]})

                input_gt = gt[0].squeeze().detach().cpu().numpy()
                recon = recon[0].squeeze().detach().cpu().numpy()
                input_gt_img = misc.get_ptcloud_img(input_gt)
                im = Image.fromarray(input_gt_img)
                gt_savepth = os.path.join(gt_pth, '%s.jpg' % model_ids[0])
                im.save(gt_savepth)
                recon_img = misc.get_ptcloud_img(recon)
                im = Image.fromarray(recon_img)
                recon_savepth = os.path.join(recon_pth, '%s.jpg' % model_ids[0])
                im.save(recon_savepth)


            for i in range(config.dataset.val.others.bs):
                _metrics = Metrics.get(dense_points[i].unsqueeze(0), gt[i].unsqueeze(0))
                if args.distributed:
                    _metrics = [dist_utils.reduce_tensor(_metric, args).item() for _metric in _metrics]
                else:
                    _metrics = [_metric.item() for _metric in _metrics]

                for _taxonomy_id in taxonomy_ids[i]:
                    if _taxonomy_id not in category_metrics:
                        category_metrics[_taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[_taxonomy_id].update(_metrics)

        
                if (idx+1) % interval == 0:
                    print_log('Test[%d/%d] Losses = %s Metrics = %s' %
                                (idx + 1, n_samples, '%.4f' % (test_loss_sum.val()*1000), 
                                ['%.4f' % m for m in _metrics]), logger=logger)

            # break
       
        #遍历字典
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())

       #输出的平均值不是总体的平均值，而是每个类的平均值的平均值，好像就是论文中的表格那样计算方式
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % test_metrics.avg(i) for i in range(4)]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    #使用wandb track重建的总loss和cdl1
    if args.local_rank == 0:
        wandb.log({'test_loss_sum': test_loss_sum.avg()*1000, 
                'test_CDL1': test_metrics.avg()[1]})

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
    if args.local_rank == 0:
        wandb.log({'test_table': table})


    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_loss_list = AverageMeter(['cdc', 'cd1', 'cd2', 'cd3', 'partial_matching'])
    test_loss_sum = AverageMeter()

    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(tqdm(test_dataloader)):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if  'PCN' in dataset_name or 'ProjectShapeNet' in dataset_name:
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial, gt)
                dense_points = ret[-1]
                recon = copy.deepcopy(dense_points)

                loss_sum, loss_list, gt_fps_list = base_model.get_loss(ret, partial, gt)
                test_loss_sum.update(loss_sum)
                test_loss_list.update(loss_list)

                _metrics = Metrics.get(dense_points, gt, require_emd=True)
                
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif 'ShapeNet' in dataset_name:
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[-1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[-1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            
            #可视化生成的图像
            if idx % 500 == 0:

                test_gt_list = [gt_fps_list[i].squeeze().detach().cpu().numpy() for i in range(4)]
                test_recon_list = [ret[i].squeeze().detach().cpu().numpy() for i in range(4)]  
                if args.local_rank == 0:
                    wandb.log({'test_gt': [wandb.Object3D(i) for i in test_gt_list],
                        'test_recon': [wandb.Object3D(i) for i in test_recon_list]})


            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, '%.4f' % (test_loss_sum.val()*1000), 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('test Metrics = %s' % (['%.4f' % test_metrics.avg(i) for i in range(4)]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.5f \t' % value
    print_log(msg, logger=logger)
    return 
