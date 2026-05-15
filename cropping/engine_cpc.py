# ------------------------------------------------------------------------
# Conditional DETR for Image Cropping
# ------------------------------------------------------------------------
# Modified from ConditionalDETR (https://github.com/Atten4Vis/ConditionalDETR)
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import numpy as np
import util.misc as utils
from dataset.cpc_eval import cpc_evaluator
from dataset.flms_eval import flms_evaluator
from dataset.gaic_transforms import reverse
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

def normalized(a, axis=-1, order=2):

    l2 = torch.atleast_1d(torch.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / torch.unsqueeze(l2, axis)

def box_to_scores(results_origin, samples, estimator, clip_model, preprocess, masks):
    all_scores = []
    for i in range(len(results_origin)):
        boxes = torch.round(results_origin[i]["boxes"]).long() #x_min, y_min, x_max, y_max
        scores = []
        for count in range(boxes.shape[0]):
            y_min = boxes[count,0]
            x_min = boxes[count,1]
            y_max = boxes[count,2]
            x_max = boxes[count,3]
            cropped_sample= samples[0, :, max(x_min,0):x_max, max(y_min,0):y_max]
            image = preprocess(cropped_sample).unsqueeze(0)
            # image = cropped_sample
            with torch.no_grad():
                image_features = clip_model.module.encode_image(image)
                im_emb_arr = normalized(image_features)
                score = estimator(im_emb_arr.type(torch.cuda.FloatTensor))
            scores.append(score[0]/10)
        scores = torch.as_tensor(scores).cuda().unsqueeze(0)
        all_scores.append(scores)
    return torch.cat(all_scores)

def align(image_retrieves):
    image_retrieves = list(image_retrieves)
    shape_x = max([image_retrieve.shape[-2] for image_retrieve in image_retrieves])
    shape_y = max([image_retrieve.shape[-1] for image_retrieve in image_retrieves])
    for i in range(len(image_retrieves)):
        image_retrieve = image_retrieves[i]
        new_image_retrieve = np.zeros((image_retrieve.shape[0], 3, shape_x, shape_y))
        new_image_retrieve[:,:,0:image_retrieve.shape[-2],0:image_retrieve.shape[-1]] = image_retrieve
        image_retrieves[i] = torch.as_tensor(new_image_retrieve).unsqueeze(0)
    return torch.cat(image_retrieves, 0).cuda().float()

def train_one_epoch(model, criterion, data_loader, optimizer,
                    device, epoch, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    iterid = 0

    if (epoch >= args.start_ema or args.force_ema) and not args.ema_initialzed:
        model.reset_moving_average()
        model.netT = model._get_target_network()
        args.ema_initialzed = 1
        model.netT.eval()
        if epoch > args.start_ema:
            print('currently we do not support resume from checkpoints whose ema has started')
            raise Exception('repeated loading of teacher')

    for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples_retrieve = samples.tensors[:,3:6].cuda()
        samples = samples.tensors[:,0:3].cuda()
        targets = [{k: v.cuda() for k, v in t.items() if k not in ["image_id"]} for t in targets]
        
        iterid += 1

        outputs = model(samples, samples_retrieve)

        for datadict in targets:
            datadict['pseudo_label'] = None

        if epoch >= args.start_ema or args.force_ema:

            with torch.no_grad():
                model.netT.eval()
                teacher_output = model.produce_pseudo_label(samples)
                pseudo_label = teacher_output['pred_logits'].sigmoid().detach()
                for idx, datadict in enumerate(targets):
                    datadict['pseudo_label'] = pseudo_label[idx]

        loss_dict = criterion(outputs, targets, epoch)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


        # reduce losses over all GPUs for logging purposes
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
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
        if (epoch >= args.start_ema) and args.enable_ema:
            model.update_moving_average()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, num_queries, set, device, output_dir, epoch=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    evaluator = cpc_evaluator(output_dir, device=device, set=set, num_queries=num_queries)

    for samples, targets, _ in metric_logger.log_every(data_loader, 10, header):
        samples_retrieve = samples.tensors[:,3:6].cuda()
        samples = samples.tensors[:,0:3].cuda()
        targets = [{k: torch.as_tensor(v).cuda() for k, v in t.items()} for t in targets]
        
        outputs = model(samples, samples_retrieve)
        for datadict in targets:
            datadict['pseudo_label'] = None

        loss_dict = criterion(outputs, targets, epoch)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results_origin, results_oriorder = postprocessors['bbox'](outputs, orig_target_sizes)
        results = results_origin

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if evaluator is not None:
            evaluator.update(batchresults=res, batchgt=targets)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if evaluator is not None:
        evaluator.evaluate()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, evaluator

def normalize(train_data, dim):
    # mean_value = train_data.mean(dim=dim, keepdim=True)
    # std_value = train_data.std(dim=dim, keepdim=True)+1e-5
    # normalized_data = (train_data - mean_value) / std_value
    min_value = torch.min(train_data)
    max_value = torch.max(train_data)+1e-5
    normalized_data = (train_data - min_value) / (max_value - min_value)
    return normalized_data

@torch.no_grad()
def evaluate_flms(model, criterion, postprocessors, data_loader, set, pathlist,
             device, output_dir, epoch, estimator, clip_model, preprocess):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    cls_threshes = [0]

    evaluator = flms_evaluator(output_dir, iou_thresh=0.85, pathlist=pathlist, set=set, cls_threshes=cls_threshes)

    for samples, targets,_, image_retrieve, embedding in data_loader:
        sam_embedding = torch.as_tensor(np.asarray(list(embedding))).cuda().float()
        image_retrieve = torch.as_tensor(np.asarray(list(image_retrieve))).cuda().float()
        # sam_embedding =None
        # image_retrieve =None
        # image_retrieve = align(image_retrieve)
        masks = samples.mask.cuda()
        samples = samples.tensors[:,0:3].cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        # targets = convert_boxes(targets, masks)
        
        # ratios = [t["ratios_retrieve"].unsqueeze(0) for t in targets]
        # box_retrieve = torch.cat(ratios).cuda().float()
        
        outputs = model(samples,image_retrieve,None, sam_embedding, masks)
        for datadict in targets:
            datadict['pseudo_label'] = None

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # orig_target_sizes = torch.stack([torch.as_tensor([1024,1024]).cuda() for t in targets], dim=0)

        # replace with targets
        # outputs['pred_boxes'][0,0:targets[0]["good_boxes"].shape[0],:] = targets[0]["good_boxes"]
        

        results_origin,_ = postprocessors['bbox'](outputs, orig_target_sizes)
        
        # scores = box_to_scores(results_origin, samples, estimator, clip_model, preprocess, masks)
        
        results = results_origin
        # rerank with predicted scores
        # results[0]["scores"] = scores[0]
        # topk_values, topk_indexes = torch.topk(results[0]["scores"], 90)
        # results[0]["scores"] = topk_values
        # results[0]["boxes"] = results[0]["boxes"][topk_indexes]
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if evaluator is not None:
            evaluator.update(batchresults=res, batchgt=targets)

    if evaluator is not None:
        evaluator.evaluate()


    return evaluator