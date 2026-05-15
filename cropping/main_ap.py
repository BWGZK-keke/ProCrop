# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------
from util.box_ops import box_xyxy_to_cxcywh
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from segment_anything import sam_model_registry
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import dataset
import util.misc as utils
from dataset.gaic import build_gaic
from dataset.flms import build_flms
from engine import train_one_epoch
from models import build_model
from collections import OrderedDict
import torch.distributed as dist
from engine_cpc import evaluate_flms
from asthetic_score import MLP
from models.conditional_detr import build
import clip
import math
import copy
from engine import update_annotations_crop
def cyclic_learning_rate(global_step,learning_rate=1e-6,max_lr=1e-3,step_size=100.,gamma=0.99994,mode='triangular',name=None):
    # cycle = math.floor( 1 + global_step / ( 2 * step_size ) )
    # x = abs( global_step / step_size - 2 * cycle + 1 )
    # clr = learning_rate + ( max_lr - learning_rate ) * max( 0, 1 - x )
    clr = max_lr * (1.0 - global_step / 500000) ** 0.9
    try:
        clr = float(clr)
    except:
        clr = float(clr.real())
    return float(clr)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-7, type=float)
    parser.add_argument('--batch_size', default=36, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=160, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--start_ema', default=100, type=int)
    parser.add_argument('--force_ema', default=0, type=int)
    parser.add_argument('--moving_average_decay', default=0.5, type=float)
    parser.add_argument('--ema_initialzed', default=0, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=90, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2, type=float)
    parser.add_argument('--soft_iou_thresh', default=0.85, type=float,
                        help='only iou with a gt crop is bigger than this, it can use the score of the gt')
    parser.add_argument('--soft_bound', default=0.5, type=float,
                        help='for redudant queries, their soft label should not bigger than this at the good dimension,'
                             'set this to 0 means do not use soft label')
    parser.add_argument('--use_valid_smooth', default=0, type=int)

    # dataset parameters
    parser.add_argument('--gpu', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--dataset_root', default='',
                        help='Root directory of the training dataset (CAD or GAIC)')
    parser.add_argument('--retrieval_db_root', default='',
                        help='Root dir for retrieval embedding databases '
                             '(subdirs: ava_self_correlated/, ava_synthetic/, ava_flms_fcdb/, ava_sacd/)')
    parser.add_argument('--ava_root', default='',
                        help='Root directory of AVA images (e.g. path/to/AVA/images/train/)')
    parser.add_argument('--sacd_root', default='',
                        help='Root directory of SACD dataset (images/ and annotations/ subdirs)')
    parser.add_argument('--fcdb_root', default='',
                        help='Root directory of FCDB dataset (images/ and annotations/ subdirs)')
    parser.add_argument('--flms_root', default='',
                        help='Root directory of FLMS dataset (image/ subdir and 500_image_dataset.mat)')
    parser.add_argument('--sam_checkpoint', default='sam_vit_b_01ec64.pth',
                        help='Path to SAM ViT-B checkpoint')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--image_size_test', default=600, type=int)
    parser.add_argument('--good_thresh', default=0, type=float)
    parser.add_argument('--good_num', default=-1, type=int)
    parser.add_argument('--nms_thresh', default=0.0, type=float)
    parser.add_argument('--topk_num', default=90, type=int)
    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--query_random_init', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='path to ConditionalDETR pretrained checkpoint', type=str)
    parser.add_argument('--resume1', default='',
                        help='path to additional pretrained checkpoint', type=str)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def crop_box(output, start_x, start_y, end_x, end_y):
    new_boxes = []
    scores= []
    for i in range(output["boxes"].shape[0]):
        if output["boxes"][i,0] >= start_x and output["boxes"][i,1]>=start_y:
            if end_x >= output["boxes"][i,2] and end_y >= output["boxes"][i,3]:
                new_boxes.append(output["boxes"][i].unsqueeze(0))
                scores.append(output["scores"][i].unsqueeze(0))
    
    if len(scores) == 0:
        return None, None
    else:
        scores = torch.cat(scores)
        boxes = torch.cat(new_boxes)
        
        boxes[:,1] = boxes[:,1] - start_y
        boxes[:,3] = boxes[:,3] - start_y
        boxes[:,0] = boxes[:,0] - start_x
        boxes[:,2] = boxes[:,2] - start_x
        return boxes, scores

def recover_box(results1, targets1,results2,targets2):
    loss_sim = 0
    count = 0
    for i in range(len(results1)):
        target1 = targets1[i]
        output1 = results1[i]
        crop_box1 = target1["crop_box"]
        target2 = targets2[i]
        output2 = results2[i]
        crop_box2 = target2["crop_box"]

        start_x = max(crop_box1[0], crop_box2[0])
        start_y = max(crop_box1[1], crop_box2[1])
        end_x = min(crop_box1[2], crop_box2[2])
        end_y = min(crop_box1[3], crop_box2[3])
        
        boxes1 = output1["boxes"].clone()
        boxes2 = output2["boxes"].clone()
        
        boxes1[:,0] = output1["boxes"][:,0] + crop_box1[0]
        boxes1[:,1] = output1["boxes"][:,1] + crop_box1[1]
        boxes1[:,2] = output1["boxes"][:,2] + crop_box1[0]
        boxes1[:,3] = output1["boxes"][:,3] + crop_box1[1]

        boxes2[:,0] = output2["boxes"][:,0] + crop_box2[0]
        boxes2[:,1] = output2["boxes"][:,1] + crop_box2[1]
        boxes2[:,2] = output2["boxes"][:,2] + crop_box2[0]
        boxes2[:,3] = output2["boxes"][:,3] + crop_box2[1]
        
        output1_crop = {}
        output1_crop["boxes"] = boxes1
        output1_crop["scores"] = output1["scores"]
        
        output2_crop = {}
        output2_crop["boxes"] = boxes2
        output2_crop["scores"] = output2["scores"]
        
        boxes1, scores1 = crop_box(output1_crop, start_x,start_y, end_x, end_y)
        boxes2, scores2 = crop_box(output2_crop, start_x,start_y, end_x, end_y)
        
        # print(output1_crop["boxes"],boxes1, start_x, start_y, end_x, end_y)
        
        if (scores1 is not None) and (scores2 is not None):
             boxes1 = box_xyxy_to_cxcywh(boxes1)
             boxes2 = box_xyxy_to_cxcywh(boxes2)
             h= end_x -start_x
             w=end_y-start_y
             boxes1 = boxes1 / torch.tensor([w, h, w, h], dtype=torch.float32).cuda()
             boxes2 = boxes2 / torch.tensor([w, h, w, h], dtype=torch.float32).cuda()
             dim = min(len(scores1), len(scores2))
             
            #  print(len(scores1), len(scores2))
             loss_sim1 = (1-F.cosine_similarity(boxes1[0:1],boxes2[0:1].detach(), dim=1))
            #  print("loss sim1:",loss_sim1.item())
            #  loss_sim2 = (1-F.cosine_similarity(scores1[0:dim].unsqueeze(0), scores2[0:dim].unsqueeze(0), dim=0)).mean()
            #  print("loss sim2:",loss_sim2.item())
             loss_sim = loss_sim + loss_sim1
             count +=1
        else:
            pass
    loss_sim = loss_sim/(count+1e-5)
    return loss_sim

        

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model, criterion, postprocessors = build_model(args)
    model, criterion, postprocessors, postprocessors_flms = build(args)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    estimator = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    estimator.load_state_dict(s)
    estimator = torch.nn.DataParallel(estimator).cuda()
    estimator.eval()
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda")  #RN50x64   
    clip_model = torch.nn.DataParallel(clip_model).cuda()
    model_without_ddp = model.module

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if ('class_embed' in n) or ('query_embed' in n) or ('decoder' in n) or ('attn' in n) or ('head' in n) and p.requires_grad]},
        # {
        #     "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        #     "lr": args.lr_backbone,
        # },        
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ((('class_embed' not in n) and ('query_embed' not in n) and ('decoder' not in n) and ('attn' not in n) and  ('head' not in n)) or ("backbone" in n )) and p.requires_grad],
            "lr": 1e-7,
        },   
    ]
    
    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },           
    # ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop, args.lr_drop + 10],
                                                        gamma=0.1)

    # dataset_train_evaluate, paths_train = build_gaic(image_set='train_evaluate', args=args)
    dataset_val_GACD, paths_val_GACD = build_gaic(image_set='val_sacd', args=args)
    # dataset_test, paths_test = build_gaic(image_set='test', args=args)
    dataset_val_flms_only, paths_val_flms_only = build_flms(image_set='val', args=args)
    dataset_val_flms, paths_val_flms = build_gaic(image_set='val_flms_fcdb', args=args)

    sampler_val_flms = torch.utils.data.SequentialSampler(dataset_val_flms)
    sampler_val_flms_only = torch.utils.data.SequentialSampler(dataset_val_flms_only)
    sampler_val_GACD = torch.utils.data.SequentialSampler(dataset_val_GACD)
    # sampler_train_evaluate = torch.utils.data.SequentialSampler(dataset_train_evaluate)
    # sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_val_flms = DataLoader(dataset_val_flms, 1, sampler=sampler_val_flms,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val_flms_only = DataLoader(dataset_val_flms_only, 1, sampler=sampler_val_flms_only,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val_GACD = DataLoader(dataset_val_GACD, 1, sampler=sampler_val_GACD,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_train_evaluate = DataLoader(dataset_train_evaluate, args.batch_size, sampler=sampler_train_evaluate,
    #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if os.path.exists(args.resume):
        # checkpoint_backbone = torch.load(args.resume, map_location='cpu')
        # pretrained_dict = checkpoint_backbone['model']
        # pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict.items() if ('class_embed' not in k)
        #                                     and ('query_embed' not in k)})
        
        checkpoint = torch.load(args.resume1, map_location='cpu')
        pretrained_dict_ours = checkpoint['model']
        # pretrained_dict_ours = OrderedDict({k: v for k, v in pretrained_dict_ours.items() if ('class_embed' not in k)
        #                                     and ('query_embed' not in k)})
        
        # for k, v in pretrained_dict.items():
        #         pretrained_dict_ours[k] = v
        
        pretrained_dict_ours = OrderedDict(pretrained_dict_ours)
        
        pretrained_dict = OrderedDict({k: v for k, v in pretrained_dict_ours.items()})
        model_dict = model_without_ddp.state_dict()
        model_dict.update(pretrained_dict)
        # print(OrderedDict(model_dict).keys())
        # model_without_ddp.conditionalDETR.load_state_dict(OrderedDict(model_dict), strict=False)
        model_without_ddp.load_state_dict(OrderedDict(model_dict), strict=False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            except:
                pass
            args.start_epoch = checkpoint['epoch'] + 1


    if args.test:

        evaluator = evaluate_flms(model, criterion, postprocessors,
                                         data_loader_val_flms, 'val', paths_val_flms,
                                         device, args.output_dir, 0, estimator, clip_model, preprocess)
        evaluator.evaluate()
        final_iou_max_dict, final_iou_mean_dict = evaluator.summary()
        evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=0)
        if args.output_dir:
            evaluator.save_cropping_results(dataset_path=args.dataset_root, epoch=0)
            utils.save_on_master(evaluator, output_dir / "test_evaluator.pth")
        return
    
    # if args.test:
    #     test_stats, evaluator = evaluate(model, criterion, postprocessors,
    #                                      data_loader_test, 'test', pathlist_val, device, args.output_dir)
    #     final_results_dict_fixed, _ = evaluator.summary(use_score_thresh=False, use_predefined_thresh=False)
    #     final_results_dict_threshed, diff_dict = evaluator.summary(use_score_thresh=True, use_predefined_thresh=True)
    #     evaluator.display_and_save_metrics(final_results_dict=final_results_dict_fixed,
    #                                        mode='topk', epoch=-1)
    #     evaluator.display_and_save_metrics(final_results_dict=final_results_dict_threshed,
    #                                        mode='thresh', epoch=-1)
    #     if args.output_dir:
    #         # evaluator.save_cropping_results(dataset_path=args.dataset_root)
    #         utils.save_on_master(evaluator, output_dir / "test_evaluator.pth")
    #     return

    print("Start training")
    start_time = time.time()
    
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam = sam.cuda()
    dataset_train, _ = build_gaic(image_set='train', args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=args.num_workers)

    
    for epoch in range(args.start_epoch, args.epochs):
        # _ = train_one_epoch(
        #     model, criterion, data_loader_train, optimizer, device, epoch,
        #     args.clip_max_norm, postprocessors, estimator, clip_model, preprocess, sam, args)
            
        model.train()
        criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 500
        iterid = 0
        count = 0
        
        # for samples, targets, images_retrieve, embedding in metric_logger.log_every(data_loader, print_freq, header):
        loader = iter(data_loader_train)
        while True:
            try:
                samples, targets, targets_aux, images_retrieve, embedding = next(loader)
            except StopIteration:
                loader = iter(data_loader_train)
                samples, targets, targets_aux, images_retrieve, embedding  = next(loader)
            
            sam_embedding = torch.as_tensor(np.asarray(list(embedding))).cuda().float()
            image_retrieve = torch.as_tensor(np.asarray(list(images_retrieve))).cuda().float()
            # samples_aux = align(samples_aux).cuda()
            # print(samples_aux.shape)
            # samples_aux = copy.deepcopy(samples)
            mask = samples.mask.cuda()
            # samples_aux = samples.tensors[:,3:6].cuda()
            # samples_aux.mask = samples_aux.mask.cuda()
            # print(samples_aux.tensors.shape, samples_aux.mask.shape)
            
            samples = samples.tensors[:,0:3].cuda()
            # samples.mask = samples.mask.cuda()
            # print(samples.tensors.shape, samples.mask.shape)
            
            
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            # targets_aux = [{k: v.cuda() for k, v in t.items()} for t in targets_aux]

            iterid += 1
            outputs = model(samples, image_retrieve, None, sam_embedding, mask)
            # outputs_aux = model(samples_aux, image_retrieve, None, sam_embedding, mask)
            
            target_sizes = torch.stack([targets[i]["orig_size"] for i in range(len(targets))], dim=0).cuda()
            results_origin, topk_results = postprocessors['bbox'](outputs, target_sizes)
            
            save_dir = os.path.join(args.dataset_root, "annotations")
            update_annotations_crop(targets, results_origin, save_dir)
            
            
            # target_sizes_aux = torch.stack([targets_aux[i]["orig_size"] for i in range(len(targets_aux))], dim=0).cuda()
            # results_origin_aux, topk_results = postprocessors['bbox'](outputs, target_sizes_aux)
            
            
            # loss_sim = recover_box(results_origin, targets,results_origin_aux,targets_aux)
            
            # for datadict in targets:
            #     datadict['pseudo_label'] = None
            # loss_dict = criterion(outputs, targets, epoch)

            try:
                loss_dict = criterion(outputs, targets, epoch)
                weight_dict = criterion.weight_dict
                
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                # losses_aux = sum(loss_dict_aux[k] * weight_dict[k] for k in loss_dict_aux.keys() if k in weight_dict)

                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

                loss_value = losses_reduced_scaled.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)

                optimizer.zero_grad()
            
                # (0.5*losses+0.5*losses_aux+loss_sim).backward()
                # (losses).backward()
                losses.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                optimizer.step()
                metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            except:
                print([target['image_id'] for target in targets])
            # loss_dict_aux = criterion(outputs_aux, targets_aux, epoch)
            
            # if iterid % 20 ==0:
            optimizer.param_groups[0]["lr"] = cyclic_learning_rate(iterid,learning_rate=1e-6,max_lr=1e-4)
            # optimizer.param_groups[1]["lr"] = cyclic_learning_rate(iterid,learning_rate=1e-9,max_lr=1e-7)

            if iterid % 100 == 0:
                # try:
                #     print("loss_sim:", loss_sim.item())
                # except:
                #     print("loss_sim:", loss_sim)
                # print("loss_aux:", losses_aux.item())
                print("losses:", losses.item())

                # print("FLMS + GACD:")
                # model.eval()
                # evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
                #                                 data_loader_val_flms, 'val', paths_val_flms,
                #                                 device, args.output_dir, epoch, estimator, clip_model, preprocess)
                # evaluator.evaluate()
                # final_iou_max_dict, final_iou_mean_dict = evaluator.summary()
                # evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)

                print("FLMS:")
                evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
                                                data_loader_val_flms_only, 'val', paths_val_flms_only,
                                                device, args.output_dir, epoch, estimator, clip_model, preprocess)
                evaluator.evaluate()
                final_iou_max_dict, final_iou_mean_dict, final_disp_max_dict = evaluator.summary()
                print(final_disp_max_dict[1].item())  
                evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)
                            
                print("GACD:")
                        
                evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
                                                data_loader_val_GACD, 'val', paths_val_GACD,
                                                device, args.output_dir, epoch, estimator, clip_model, preprocess)
                evaluator.evaluate()
                final_iou_max_dict, final_iou_mean_dict, final_disp_max_dict = evaluator.summary()
                evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)  
                print(final_disp_max_dict[1].item())  

                if args.output_dir:
                    checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
                    checkpoint_paths.append(os.path.join(output_dir, f'checkpoint{iterid:06}'+str(final_iou_max_dict[1].item())+'.pth'))
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)

        # print("FLMS + GACD:")
        # model.eval()
        # evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
        #                                  data_loader_val_flms, 'val', paths_val_flms,
        #                                  device, args.output_dir, epoch, estimator, clip_model, preprocess)
        # evaluator.evaluate()
        # final_iou_max_dict, final_iou_mean_dict = evaluator.summary()
        # evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)
        
        
        # print("FLMS:")
        # evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
        #                                  data_loader_val_flms_only, 'val', paths_val_flms_only,
        #                                  device, args.output_dir, epoch, estimator, clip_model, preprocess)
        # evaluator.evaluate()
        # final_iou_max_dict, final_iou_mean_dict = evaluator.summary()
        # evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)
        
        
        # print("GACD:")
        # evaluator = evaluate_flms(model_without_ddp.conditionalDETR, criterion, postprocessors,
        #                                  data_loader_val_GACD, 'val', paths_val_GACD,
        #                                  device, args.output_dir, epoch, estimator, clip_model, preprocess)
        # evaluator.evaluate()
        # final_iou_max_dict, final_iou_mean_dict = evaluator.summary()
        # evaluator.display_and_save_metrics(final_iou_max_dict, final_iou_mean_dict, epoch=epoch)

        # if epoch == 3:
        #     test_stats, evaluator = evaluate(model, criterion, postprocessors,
        #                                 data_loader_train_evaluate, args.num_queries,
        #                                 'train', device, args.output_dir, epoch, estimator, clip_model, preprocess)

    #     test_stats, evaluator = evaluate(model, criterion, postprocessors,
    #                                      data_loader_val, args.num_queries,
    #                                      'val', device, args.output_dir, epoch, estimator, clip_model, preprocess)

    #     log_stats = {#**{f'train_{k}': v for k, v in train_stats.items()},
    #                  **{f'test_{k}': v for k, v in test_stats.items()},
    #                  'epoch': epoch,
    #                  'n_parameters': n_parameters}

    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #     if evaluator is not None:
    #         evaluator.evaluate()

    #         if utils.is_main_process():
    #             (output_dir / 'val').mkdir(exist_ok=True)
    #             filenames = ['latest.pth']
    #             if epoch % 50 == 0:
    #                 filenames.append(f'{epoch:03}.pth')
    #             for name in filenames:
    #                 torch.save(evaluator,
    #                            output_dir / "val" / name)
    #             evaluator.display_and_save_apmetrics(iou_thresh=0.90, epoch=epoch)
    #             evaluator.display_and_save_apmetrics(iou_thresh=0.85, epoch=epoch)
    #             evaluator.display_and_save_accmetrics(iou_thresh=0.90, epoch=epoch)
    #             evaluator.display_and_save_accmetrics(iou_thresh=0.85, epoch=epoch)
    #     # del evaluator
    #     torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
    main(args)
