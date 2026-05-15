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
from scipy.ndimage import zoom
import torch
import numpy as np
import util.misc as utils
from dataset.flms_eval import flms_evaluator
from dataset.gaic_eval_ap import gaic_evaluator
from dataset.gaic_transforms import reverse
from PIL import Image
from matplotlib import cm
from scipy.stats import kendalltau
from torchmetrics.regression import KendallRankCorrCoef
import torch.nn.functional as F
from segment_anything import sam_model_registry

def cyclic_learning_rate(global_step,learning_rate=1e-6,max_lr=1e-5,step_size=500.,gamma=0.99994,mode='triangular',name=None):
  cycle = math.floor( 1 + global_step / ( 2 * step_size ) )
  x = abs( global_step / step_size - 2 * cycle + 1 )
  clr = learning_rate + ( max_lr - learning_rate ) * max( 0, 1 - x )
  return clr


class CCCLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(CCCLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_hat):
        y_true_mean = torch.mean(y_true,1, keepdim=True)
        y_hat_mean = torch.mean(y_hat,1, keepdim=True)
        y_true_var = torch.var(y_true,1, keepdim=True)
        y_hat_var = torch.var(y_hat,1, keepdim=True)
        y_true_std = torch.std(y_true,1, keepdim=True)
        y_hat_std = torch.std(y_hat,1, keepdim=True)
        vx = y_true - torch.mean(y_true,1, keepdim=True)
        vy = y_hat - torch.mean(y_hat,1, keepdim=True)
        pcc = torch.sum(vx * vy) / (torch.norm(vx) * torch.norm(vy) + self.eps)
        ccc = (2 * pcc * y_true_std * y_hat_std) / \
            (y_true_var + y_hat_var + (y_hat_mean - y_true_mean) ** 2)
        ccc = 1 - ccc
        return ccc*0.1

def normalized(a, axis=-1, order=2):

    l2 = torch.atleast_1d(torch.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / torch.unsqueeze(l2, axis)

def save_annotations(targets, scores, results_origin, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(results_origin)):
        target = targets[i]
        output = results_origin[i]
        file_name = str(target["image_id"].item())+".txt"
        file_path = os.path.join(save_dir, file_name)
        box = output["boxes"].long()
        score = scores[i].unsqueeze(-1)
        box = torch.cat([box, score],-1)
        txt = "\n".join([" ".join(str(int(x.item())) for x in box[i]) for i in range(box.shape[0])])
        # Open the file in write mode
        with open(file_path, "w") as file:
            # Write the text to the file
            file.write(txt)
            file.close()
            
def read_annotation(file_path):
    with open(file_path, 'r') as fid:
        annotations = fid.readlines()
        fid.close()
    matrix = np.zeros((len(annotations), 5))
    for i in range(len(annotations)):
        annotation_split = annotations[i].split()
        current_score = float(annotation_split[4])
        xmin = float(annotation_split[1])
        ymin = float(annotation_split[0])
        xmax = float(annotation_split[3])
        ymax = float(annotation_split[2])
        matrix[i, 0] = xmin
        matrix[i, 1] = ymin
        matrix[i, 2] = xmax
        matrix[i, 3] = ymax
        matrix[i, 4] = current_score
    return matrix
        
def remove_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# def select_diverse(boxes):
#     target_ratios = [1, 5/4, 4/3, 7/5, 3/2, 16/9]
#     selected_boxes = []
#     selected_index  = []
#     selected_size = []
#     for i in range(boxes.shape[0]):
#         ratio = (boxes[i][2]-boxes[i][0])/((boxes[i][3]-boxes[i][1])+1e-5)
#         if ratio > 0:
#             if ratio < 1:
#                 ratio = 1/ratio
#         else:
#             continue
        
#         if len(selected_boxes) == 0:
#             target_size = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
#             selected_boxes.append(boxes[i:i+1])
#             selected_index.append(i)
#             selected_size.append(target_size)
#             target_array = np.zeros((int(selected_boxes[0][0][2])+600, int(selected_boxes[0][0][3])+600))
#             target_array[int(selected_boxes[0][0][0]):int(selected_boxes[0][0][2]), int(selected_boxes[0][0][1]):int(selected_boxes[0][0][3])] = 1
#             cover = target_array.sum()
#         else:
#             current_size = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
#             ratio_difference = [abs(ratio-i) for i in target_ratios]    
#             if min(ratio_difference) <0.02:
#                 ratio_index = ratio_difference.index(min(ratio_difference))
#                 target_ratio = target_ratios[ratio_index]
#                 if ratio <1:
#                     boxes[i,2] = int((boxes[i][3]-boxes[i][1])/target_ratio)+boxes[i, 0]  
#                 else:
#                     boxes[i,2] = int(target_ratio*(boxes[i][3]-boxes[i][1]))+boxes[i, 0]  
                
#             size_difference = [abs(current_size - s)/target_size for s in selected_size]
#             pos_difference = [abs(boxes[i][0]-s[0][0])/(boxes[0][2]-boxes[0][0])+abs(boxes[i][1]-s[0][1])/(boxes[0][3]-boxes[0][1])+abs(boxes[i][2]-s[0][2])/(boxes[0][2]-boxes[0][0])+abs(boxes[i][3]-s[0][3])/(boxes[0][3]-boxes[0][1]) for s in selected_boxes]
            
            
#             current_array = np.zeros((int(selected_boxes[0][0][2])+600, int(selected_boxes[0][0][3])+600))
#             current_array[int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])] = 1
#             inter = (current_array*target_array).sum()
                
#             if min(size_difference+pos_difference) > 1 and inter/cover > 0.8:
#                 selected_boxes.append(boxes[i:i+1])
#                 selected_index.append(i)   
#                 selected_size.append(current_size)
                    
#     return np.concatenate(selected_boxes)


def select_diverse(boxes,img_shape):
    img_size = img_shape[0]*img_shape[1]
    target_ratios = [1, 5/4, 4/3, 7/5, 3/2, 16/9]
    selected_boxes = []
    selected_index  = []
    selected_size = []
    for i in range(boxes.shape[0]):
        ratio = (boxes[i][2]-boxes[i][0])/((boxes[i][3]-boxes[i][1])+1e-5)
        if ratio > 0:
            if ratio < 1:
                ratio = 1/ratio
        
        if len(selected_boxes) == 0:
            target_size = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
            selected_boxes.append(boxes[i:i+1])
            selected_index.append(i)
            selected_size.append(target_size)
            target_array = np.zeros((int(selected_boxes[0][0][2])+600, int(selected_boxes[0][0][3])+600))
            target_array[int(selected_boxes[0][0][0]):int(selected_boxes[0][0][2]), int(selected_boxes[0][0][1]):int(selected_boxes[0][0][3])] = 1
            cover = target_array.sum()
            
        else:
            current_size = (boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
            ratio_difference = [abs(ratio-i) for i in target_ratios]    
            if min(ratio_difference) <0.03:
                ratio_index = ratio_difference.index(min(ratio_difference))
                target_ratio = target_ratios[ratio_index]
                if ratio <1:
                    boxes[i,2] = int((boxes[i][3]-boxes[i][1])/target_ratio) +boxes[i, 0]  
                else:
                    boxes[i,2] = int(target_ratio*(boxes[i][3]-boxes[i][1]))+boxes[i, 0]  
                
                size_difference = [abs(current_size - s)/img_size for s in selected_size]
                pos_difference = [abs(boxes[i][0]-s[0][0])/img_shape[0]+abs(boxes[i][1]-s[0][1])/img_shape[1]+abs(boxes[i][2]-s[0][2])/img_shape[0]+abs(boxes[i][3]-s[0][3])/img_shape[1] for s in selected_boxes]
                if min(2*size_difference+pos_difference) > 0.2:
                    current_array = np.zeros((int(selected_boxes[0][0][2])+600, int(selected_boxes[0][0][3])+600))
                    current_array[int(boxes[i][0]):int(boxes[i][2]), int(boxes[i][1]):int(boxes[i][3])] = 1
                    inter = (current_array*target_array).sum()
                    if inter/cover > 0.8:
                        selected_boxes.append(boxes[i:i+1])
                        selected_index.append(i)   
                        selected_size.append(current_size)
                        
    return np.concatenate(selected_boxes)

# def update_annotations(targets, results_origin, results_origin_score, samples, estimator, clip_model, preprocess, masks):
#     save_dir = "/home/zhangke/synthetic_dataset_v8/annotations/"
#     for i in range(len(results_origin)):
#         target = targets[i]
#         output = results_origin[i]
#         output_score = results_origin_score[i]
#         file_name = str(target["image_id"].item())+".txt"
#         file_path = os.path.join(save_dir, file_name)
#         annotation_ori = read_annotation(file_path)
#         pseudo_label = annotation_ori[0:1]
#         selected_box, selected_index = select_diverse(output["boxes"], 50)
#         boxes = torch.cat([output_score["boxes"][i:i+1] for i in selected_index])
#         score = box_to_score(boxes, samples, estimator, clip_model, preprocess, masks)
#         score = score[0].unsqueeze(-1)*10
#         box = selected_box.long()
#         # score = scores[i].unsqueeze(-1)*10
#         box = torch.cat([box, score],-1).detach().cpu().numpy()
        
#         all_annotations = np.concatenate([box, annotation_ori[1:]])
#         # all_annotations = box
        
#         all_scores = all_annotations[:,-1]
        
#         box_thresh = all_annotations[all_scores > pseudo_label[:,-1]]
#         score_threshold = all_scores[all_scores > pseudo_label[:,-1]]
#         if len(score_threshold) > 0:      
#             scores_index = np.argsort(score_threshold)[::-1]
#             selected_annotations = box_thresh[scores_index][0:4]
#             # print(selected_annotations.shape, selected_annotations)
#             selected_annotations = np.concatenate([pseudo_label, selected_annotations])
#         else:
#             selected_annotations = pseudo_label
            
#         txt = "\n".join(remove_duplicate([" ".join(str(max(x.item(),0)) for x in selected_annotations[i]) for i in range(selected_annotations.shape[0])]))
#         with open(file_path, "w+") as file:
#             file.write(txt)
#             file.close()

def get_crop_size(targets):
    crop_size = []
    for target in targets:
        crop_box = target["crop_box"]
        start_x = crop_box[0]
        start_y = crop_box[1]
        end_x = crop_box[2]
        end_y = crop_box[3]
        w = end_x - start_x
        h = end_y - start_y
        crop_size.append((w,h))
    crop_size = torch.tensor(crop_size)
    return crop_size.cuda()
            

def update_annotations_crop(targets, results_origin, save_dir):
    for i in range(len(results_origin)):
        
        target = targets[i]
        output = results_origin[i]
        output_score = output["scores"].detach()
        
        file_name = str(target["image_id"].item())+".txt"
        crop_box = target["crop_box"]
        
        predict_box = output["boxes"].detach()
        
        start_x = crop_box[0]
        start_y = crop_box[1]
        
        predict_box[:,0] = predict_box[:,0] + start_x
        predict_box[:,1] = predict_box[:,1] + start_y
        predict_box[:,2] = predict_box[:,2] + start_x
        predict_box[:,3] = predict_box[:,3] + start_y
        
        file_path = os.path.join(save_dir, file_name)
        annotation_ori = read_annotation(file_path)
        
        pseudo_label = annotation_ori[0:1]
        score = output_score.unsqueeze(-1)*10
        box = predict_box.long()
        box = torch.cat([box, score.cuda()],-1).cpu().numpy()      
        all_annotations = np.concatenate([annotation_ori, box])

        all_annotations = select_diverse(all_annotations,(crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]))
        all_annotations = all_annotations[1:]
        
        all_scores = all_annotations[:,-1]
        
        box_thresh = all_annotations[all_scores > 2]
        score_threshold = all_scores[all_scores > 2]
        
        # box_thresh = all_annotations
        # score_threshold = all_scores
        if len(score_threshold) > 0:
            write_txt = []      
            scores_index = np.argsort(score_threshold)[::-1]
            selected_annotations = box_thresh[scores_index][0:4]
            selected_annotations = np.concatenate([pseudo_label, selected_annotations])
            for i in range(selected_annotations.shape[0]):
                ann = selected_annotations[i]
                ann[0] = max(selected_annotations[i][0].item(),0)
                ann[1] = max(selected_annotations[i][1].item(),0)
                ann[2] = min(selected_annotations[i][2].item(), target["orig_size"][0].item())
                ann[3] = min(selected_annotations[i][3].item(), target["orig_size"][1].item())
                ann[4] = selected_annotations[i][4].item()
                ann_str = " ".join([str(ann[1]), str(ann[0]), str(ann[3]), str(ann[2]), str(ann[4])])
                write_txt.append(ann_str)
            write_txt = "\n".join(remove_duplicate(write_txt))
            with open(file_path, "w+") as file:
                file.write(write_txt)
                file.close()
        else:
            pass            


        
        
def convert_mask_to_size(masks):
    sizes = []
    for i in range(masks.shape[0]):
        mask = torch.where(masks[i] == False, 1, 0)
        coordinates = torch.nonzero(mask)
        xl,xr = coordinates[:,0].min(),coordinates[:,0].max()
        yl,yr = coordinates[:,1].min(),coordinates[:,1].max()
        dx = xr+1-xl
        dy = yr+1-yl
        size = torch.as_tensor((dx, dy))
        sizes.append(size.unsqueeze(0))
    return torch.cat(sizes)
    
def box_to_scores(results_origin, samples, estimator, clip_model, preprocess, masks):
    all_scores = []
    for i in range(len(results_origin)):
        boxes = torch.round(results_origin[i]["boxes"]).long() #x_min, y_min, x_max, y_max
        scores = []
        for count in range(boxes.shape[0]):
        # for count in range(20):
            y_min = boxes[count,0]
            x_min = boxes[count,1]
            y_max = boxes[count,2]
            x_max = boxes[count,3]
            cropped_sample= samples[0, :, max(x_min,0):max(x_max, max(x_min,0)+1), max(y_min,0): max(y_max,max(y_min,0)+1)]
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

def box_to_scores_single(results_origin, samples, estimator, clip_model, preprocess, masks):
    all_scores = []
    selected_indexes = []
    for i in range(len(results_origin)):
        boxes = torch.round(results_origin[i]["boxes"]).long() #x_min, y_min, x_max, y_max
        scores_pred = results_origin[i]["scores"]
        boxes_new, selected_index = select_diverse(boxes.detach().cpu().numpy(), 20, True)
        if boxes_new.shape[0]>1:
            boxes = boxes_new
            k = boxes.shape[0]
            selected_indexes.append(selected_index)
        else:
            k = 1
            selected_index = [0]
            selected_indexes.append(selected_index)
        scores = []
        for count in range(k):
        # for count in range(20):
            # y_min = boxes[count,0]
            # x_min = boxes[count,1]
            # y_max = boxes[count,2]
            # x_max = boxes[count,3]
            # cropped_sample= samples[0, :, max(x_min,0):max(x_max, max(x_min,0)+1), max(y_min,0): max(y_max,max(y_min,0)+1)]
            # image = preprocess(cropped_sample).unsqueeze(0)
            # selected_index[count]
            # # image = cropped_sample
            # with torch.no_grad():
            #     image_features = clip_model.module.encode_image(image)
            #     im_emb_arr = normalized(image_features)
            #     score = estimator(im_emb_arr.type(torch.cuda.FloatTensor))
            # scores.append(score[0]/10)
            scores.append(scores_pred[selected_index[count]])
        scores = torch.as_tensor(scores).cuda().unsqueeze(0)
        all_scores.append(scores)
    return selected_indexes, all_scores


def box_to_embedding(results_origin, samples, sam):
    all_scores = []
    all_embeddings = []
    selected_indexes = []
    for i in range(len(results_origin)):
        boxes = torch.round(results_origin[i]["boxes"]).long() #x_min, y_min, x_max, y_max
        scores_pred = results_origin[i]["scores"]
        boxes_new, selected_index = select_diverse(boxes.detach().cpu().numpy(), 20, True)
        if boxes_new.shape[0]>1:
            boxes = boxes_new
            k = boxes.shape[0]
            selected_indexes.append(selected_index)
        else:
            k = 1
            selected_index = [0]
            selected_indexes.append(selected_index)
        scores = []
        embeddings = []
        for count in range(k):
            # y_min = boxes[count,0]
            # x_min = boxes[count,1]
            # y_max = boxes[count,2]
            # x_max = boxes[count,3]
            #cropped_sample= samples[0, :, max(x_min,0):max(x_max, max(x_min,0)+1), max(y_min,0): max(y_max,max(y_min,0)+1)]
            # with torch.no_grad():
            #     embedding = extract_feature(cropped_sample, sam)
            #     embeddings.append(embedding)
            scores.append(scores_pred[selected_index[count]])
        # all_embeddings.append(embeddings)
        scores = torch.as_tensor(scores).cuda().unsqueeze(0)
        all_scores.append(scores)
    return selected_indexes, all_scores#, all_embeddings


def box_to_score(boxes, samples, estimator, clip_model, preprocess, masks):
    boxes = torch.round(boxes).long() #x_min, y_min, x_max, y_max
    scores = []
    for count in range(boxes.shape[0]):
        y_min = boxes[count,0]
        x_min = boxes[count,1]
        y_max = boxes[count,2]
        x_max = boxes[count,3]
        cropped_sample= samples[0, :, max(x_min,0):max(x_max, max(x_min,0)+1), max(y_min,0): max(y_max,max(y_min,0)+1)]
        image = preprocess(cropped_sample).unsqueeze(0)
        # image = cropped_sample
        with torch.no_grad():
            image_features = clip_model.module.encode_image(image)
            im_emb_arr = normalized(image_features)
            score = estimator(im_emb_arr.type(torch.cuda.FloatTensor))
        scores.append(score[0]/10)
    scores = torch.as_tensor(scores).cuda().unsqueeze(0)
    return scores


# def convert_boxes(targets, masks): #[w, h, w, h]
#     for i in range(len(targets)):
#         target = targets[i]
#         boxes = target["good_boxes"]
        
#         mask = torch.where(masks[i] == False, 1, 0)
#         coordinates = torch.nonzero(mask)
#         xl,xr = coordinates[:,1].min(),coordinates[:,1].max()
#         yl,yr = coordinates[:,0].min(),coordinates[:,0].max()
#         dx = xr+1-xl
#         dy = yr+1-yl
#         dx_ori = masks.shape[2]
#         dy_ori = masks.shape[1]
        
#         boxes[:,0] = boxes[:,0]#*dx/dx_ori 
#         boxes[:,1] = boxes[:,1]#*dy/dy_ori
#         boxes[:,2] = boxes[:,2]#*dx/dx_ori
#         boxes[:,3] = boxes[:,3]#*dy/dy_ori
        
#         target["good_boxes"] = boxes.cuda()
#         target["ratios_retrieve"][:,0] = target["ratios_retrieve"][:,0]#*dx/dx_ori
#         target["ratios_retrieve"][:,1] = target["ratios_retrieve"][:,1]#*dy/dy_ori
#         target["ratios_retrieve"] = target["ratios_retrieve"].cuda()
#         targets[i] = target
        
#     return targets

def convert_boxes(targets, masks): #[w, h, w, h]
    for i in range(len(targets)):
        target = targets[i]
        for key in ["good_boxes", "boxes"]:
            boxes = target[key]
            
            mask = torch.where(masks[i] == False, 1, 0)
            coordinates = torch.nonzero(mask)
            xl,xr = coordinates[:,1].min(),coordinates[:,1].max()
            yl,yr = coordinates[:,0].min(),coordinates[:,0].max()
            dx = xr+1-xl
            dy = yr+1-yl
            dx_ori = masks.shape[2]
            dy_ori = masks.shape[1]
            
            boxes[:,0] = boxes[:,0]*dx/dx_ori 
            boxes[:,1] = boxes[:,1]*dy/dy_ori
            boxes[:,2] = boxes[:,2]*dx/dx_ori
            boxes[:,3] = boxes[:,3]*dy/dy_ori
            
            target[key] = boxes.cuda()
            targets[i] = target
    return targets

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


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
        # sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        # sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        # positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(1 / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

def normalize(train_data):
    # mean_value = train_data.mean(dim=dim, keepdim=True)
    # std_value = train_data.std(dim=dim, keepdim=True)+1e-5
    # normalized_data = (train_data - mean_value) / std_value
    # # return normalized_data
    # print(train_data.min(), train_data.max(), train_data)
    min_value = torch.min(train_data)
    max_value = torch.max(train_data)+1e-5
    normalized_data = (train_data - min_value) / (max_value - min_value)
    return normalized_data

def extract_feature(image, sam):
    samples_new = zoom(image.detach().cpu().numpy(), (1, 1024/image.shape[-2] , 1024 /image.shape[-1]), order=0)
    samples_new = torch.tensor(samples_new).unsqueeze(0)
    embedding = sam.image_encoder(samples_new[0:1].cuda()).detach().cpu().numpy()
    embedding = zoom(embedding, (1,0.25,0.25,0.25)).flatten()
    return torch.as_tensor(embedding)

def compare_embedding(crop, retrieve):
    retrieve_flat = retrieve.flatten(2)
    indexes = []
    all_similarities = []
    for i in range(len(crop)):
        crop_i = crop[i]
        similarities = []
        for j in range(len(crop_i)):
            # print(crop_i[j].unsqueeze(0).shape, retrieve_flat[i].shape)
            similarity = F.cosine_similarity(crop_i[j].unsqueeze(0).cuda(), retrieve_flat[i], dim=1)
            similarities.append(similarity.max())
        all_similarities.append(similarities)
    return all_similarities

def train_one_epoch(model, criterion,data_loader, optimizer, device, epoch, max_norm, postprocessors, estimator, clip_model, preprocess,sam, args):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500
    iterid = 0
    count = 0
    
    
    # if (epoch >= args.start_ema or args.force_ema) and not args.ema_initialzed:
    #     model.module.reset_moving_average()
    #     model.module.netT = model.module._get_target_network()
    #     args.ema_initialzed = 1
    #     model.module.netT.eval()
    #     if epoch > args.start_ema:
    #         print('currently we do not support resume from checkpoints whose ema has started')
    #         raise Exception('repeated loading of teacher')

    # for samples, targets, images_retrieve, embedding in metric_logger.log_every(data_loader, print_freq, header):
    loader = iter(data_loader)
    while True:
        try:
            samples, targets, targets_aux, images_retrieve, embedding = loader.next()
        except StopIteration:
            loader = iter(data_loader)
            samples, targets, targets_aux, images_retrieve, embedding  = loader.next()
        sam_embedding = torch.as_tensor(np.asarray(list(embedding))).cuda().float()
        image_retrieve = torch.as_tensor(np.asarray(list(images_retrieve))).cuda().float()

        samples = samples.tensors[:,0:3].cuda()
        samples_aux = samples.tensors[:,3:6].cuda()
        
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        targets_aux = [{k: v.cuda() for k, v in t.items()} for t in targets_aux]

        iterid += 1
        # outputs = model(samples, image_retrieve, box_retrieve, sam_embedding)
        outputs = model(samples, image_retrieve, None, sam_embedding)
        outputs_aux = model(samples_aux, image_retrieve, None, sam_embedding)
        
        loss_dict = criterion(outputs, targets, epoch)
        loss_dict_aux = criterion(outputs_aux, targets_aux, epoch)
        
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_aux = sum(loss_dict_aux[k] * weight_dict[k] for k in loss_dict_aux.keys() if k in weight_dict)


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
    
        (losses+losses_aux).backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if (epoch >= args.start_ema):
            model.module.update_moving_average()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if iterid % 20 ==0:
            optimizer.param_groups[0]["lr"] = cyclic_learning_rate(iterid)

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, num_queries, set, device, output_dir, epoch, estimator, clip_model, preprocess):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = gaic_evaluator(output_dir, device=device, set="val", num_queries=num_queries)

    for samples, targets, image_retrieve, embedding in metric_logger.log_every(data_loader, 10, header):
        sam_embedding = normalize(torch.as_tensor(list(embedding)).cuda().float())
        image_retrieve = normalize(torch.as_tensor(np.asarray(list(image_retrieve))).cuda().float())
        # image_retrieve = align(image_retrieve)
        masks = samples.mask.cuda()
        samples = samples.tensors.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        # targets = convert_boxes(targets, masks)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        # ratios = [t["ratios_retrieve"].unsqueeze(0) for t in targets]
        # box_retrieve = torch.cat(ratios).cuda().float()
        
        # outputs = model(samples, image_retrieve, box_retrieve, sam_embedding)
        outputs = model(samples, image_retrieve, None, sam_embedding)
        # target_sizes = torch.stack([torch.as_tensor([samples.shape[2], samples.shape[3]])], dim=0).cuda()
        target_sizes = convert_mask_to_size(masks).cuda()
        target_sizes_2 = torch.stack([targets[i]["orig_size"] for i in range(len(targets))], dim=0).cuda()


        results_origin, topk_results = postprocessors['bbox'](outputs, target_sizes) #x_min, y_min, x_max, y_max
        results_origin_2, topk_results = postprocessors['bbox'](outputs, target_sizes_2)

        # topk_values, topk_indices = topk_results
        # scores = box_to_scores(results_origin, samples, estimator, clip_model, preprocess, masks)
        
        if set == "train":
            # save_annotations(targets, scores, results_origin)
            update_annotations(targets, results_origin_2, results_origin, samples, estimator, clip_model, preprocess, masks)

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

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if evaluator is not None:
        evaluator.evaluate()


    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats, evaluator


@torch.no_grad()
def evaluate_single(model, postprocessors, data_loader, device):
    model.eval()

    res = {}
    for samples, targets in data_loader:
        samples = samples.tensors.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        outputs = model(samples)


        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results_origin = postprocessors['bbox'](outputs, orig_target_sizes)
        results = results_origin

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}


    return res

@torch.no_grad()
def evaluate_flms(model, criterion, postprocessors, data_loader, set, pathlist,
             device, output_dir, epoch=0):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    cls_threshes = [0]

    evaluator = flms_evaluator(output_dir, iou_thresh=0.85, pathlist=pathlist, set=set, cls_threshes=cls_threshes)

    for samples, targets, image_retrieve in data_loader:
        image_retrieve = align(image_retrieve)
        masks = samples.mask.cuda()
        samples = samples.tensors.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        # targets = convert_boxes(targets, masks)
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        ratios = [t["ratios_retrieve"].unsqueeze(0) for t in targets]
        box_retrieve = torch.cat(ratios).cuda().float()
        
        outputs = model(samples, image_retrieve, box_retrieve)
        for datadict in targets:
            datadict['pseudo_label'] = None

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results_origin, results_oriorder = postprocessors['bbox'](outputs, orig_target_sizes)
        results = results_origin

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if evaluator is not None:
            evaluator.update(batchresults=res, batchgt=targets)

    if evaluator is not None:
        evaluator.evaluate()


    return evaluator