import sys
import os
import torch.utils.data as data
import cv2
import math
import numpy as np
import random
import torch
import datetime
from dataset.gaic_transforms import croptransform, reverse
from util.box_ops import box_iou
from torchvision.ops import nms
import datasets as ds
import fsspec
import pandas as pd
import warnings
import copy

MOS_MEAN = 2.95
MOS_STD = 0.8
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

def find_data_files(data_dir, splits):
    """Find data files in each split."""
    splits = splits or ["train", "val", "test"]
    def _glob(url_prefix: str) -> list[str]:
        fs, path_prefix = fsspec.core.url_to_fs(url_prefix)
        return fs.glob(path_prefix + "*")  # type: ignore

    data_files = {split: _glob(os.path.join(data_dir, split)) for split in splits}
    if any(len(x) == 0 for x in data_files.values()):
        raise FileNotFoundError(f"No dataset file found at {data_dir} with {splits}")
    return data_files


def prepare_original_data(image, annofile, thresh, good_num=-1, nms_thresh=0.8, set = ""):
    # image: image read from cv2, therefore, it is numpy array
    # annofile: coordinates and the corresponding scores loaded from files
    # thresh: score thresh to define good crops
    # the output is not processed since augmentation needs original format
    # the output image is BGR and unit8 (0-255) numpy of original size
    # the output target score range is 1-5
    # the output target boxes is [xmin, ymin, xmax, ymax] of original size
    # if there is no crop whose score is higher than the threshold
    # we will use the crops with the highest score
    if thresh * good_num > 0:
        raise Exception('thresh and good_num cannot be both positive or both negative')

    bbox = list()
    score = list()

    good_score = list()
    good_bbox = list()

    for i in range(len(annofile)):
        annotation = annofile[i]
        annotation_split = annotation.split()
        # current_score = float(annotation_split[4])
        if len(annotation_split) == 4:
            current_score = 5
        else:
            try:
                current_score = float(annotation_split[4])
            except:
                current_score = float(annotation_split[4][0:5])
        
        # For SACD dataset
        # if "sacd" in set:
        #     xmin = max(float(annotation_split[0]),0)
        #     ymin = max(float(annotation_split[1]),0)
        #     xmax = max(float(annotation_split[2]),xmin+1)
        #     ymax = max(float(annotation_split[3]),ymin+1)
        # else:
        #     xmin = max(float(annotation_split[1]),0)
        #     ymin = max(float(annotation_split[0]),0)
        #     xmax = max(float(annotation_split[3]),xmin+1)
        #     ymax = max(float(annotation_split[2]),ymin+1)

            # print("sacd:", annotation_split, image.shape)
        xmin = max(float(annotation_split[1]),0)
        ymin = max(float(annotation_split[0]),0)
        xmax = min(float(annotation_split[3]), image.shape[0])
        ymax = min(float(annotation_split[2]), image.shape[1])
        if xmax > image.shape[0] or ymax > image.shape[1]:
            print("generated_data:",i,xmax,ymax)
            print(annofile[0].split()[3],annofile[0].split()[2], image.shape)

        if current_score != -2:

            bbox.append([xmin, ymin, xmax, ymax])
            score.append(current_score)

            if thresh > 0:
                if current_score >= thresh:
                    good_bbox.append([xmin, ymin, xmax, ymax])
                    good_score.append(current_score)


    bbox = torch.as_tensor(bbox)
    # iou_mat = box_iou(bbox, bbox)[0]
    # torch.set_printoptions(edgeitems=100)
    # print(iou_mat)
    # exit()
    score = torch.as_tensor(score)

    if good_num > 0:
        good_score, indices = torch.sort(score, descending=True)
        good_score = good_score[:good_num]
        good_bbox = bbox[indices[:good_num]]
    else:
        good_bbox = torch.as_tensor(good_bbox)
        good_score = torch.as_tensor(good_score)
        

    if good_bbox.shape[0] == 0:
        best_score = torch.max(score)
        max_index = (score == best_score).nonzero()
        best_bbox = bbox[max_index]
        bestnum = max_index.shape[0]
        best_bbox = best_bbox.view(bestnum, -1).contiguous()
        best_score = best_score.unsqueeze(0).repeat(bestnum)
        good_bbox = best_bbox
        good_score = best_score

    if nms_thresh > 0:
        # there may be some very close crops with high scores
        # we only keep the one with the highest score
        id_remained = nms(good_bbox, good_score, iou_threshold=nms_thresh)
        good_bbox = good_bbox[id_remained]
        good_score = good_score[id_remained]


    orig_size = torch.as_tensor(list(image.shape[:2]))
    label = torch.zeros_like(good_score).type(torch.LongTensor)

    target = {'boxes': bbox, 'good_boxes': good_bbox,
              'scores': score, 'good_scores': good_score,
              'orig_size': orig_size, 'labels':label,}
              # 'orig_good_boxes': good_bbox}

    return image, target



class GAICD(data.Dataset):

    def __init__(self, args, set, imgpath=None, annopath=None, transform=None):

        self._imgpath = imgpath
        self._annopath = annopath
        self.set = set
        self.good_thresh = args.good_thresh
        self.transform = transform
        self.good_num = args.good_num
        self.nms_thresh = args.nms_thresh
        self.ava_root = args.ava_root

        # Load pre-extracted SAM embedding databases (built by calculate_retrieval_relationships/)
        # retrieval_db_root should contain subdirs: ava_self_correlated/, ava_synthetic/,
        # ava_flms_fcdb/, ava_sacd/
        db_root = args.retrieval_db_root
        ava_files = find_data_files(
            os.path.join(db_root, "ava_self_correlated"),
            splits=["train"],
        )
        synthetic_files = find_data_files(
            os.path.join(db_root, "ava_synthetic"),
            splits=["train"],
        )
        flms_fcdb_files = find_data_files(
            os.path.join(db_root, "ava_flms_fcdb"),
            splits=["train"],
        )
        sacd_files = find_data_files(
            os.path.join(db_root, "ava_sacd"),
            splits=["train"],
        )
        self.dataset = ds.load_dataset("parquet", data_files=ava_files)
        self.flms_fcdb_dataset = ds.load_dataset("parquet", data_files=flms_fcdb_files)
        self.sacd_dataset = ds.load_dataset("parquet", data_files=sacd_files)
        self.synthetic_dataset = ds.load_dataset("parquet", data_files=synthetic_files)
        self.ids = self.dataset["train"]["id"]
        self.flms_fcdb_ids = self.flms_fcdb_dataset["train"]["annotation_name"]
        self.sacd_ids = self.sacd_dataset["train"]["annotation_name"]
        self.synthetic_ids = self.synthetic_dataset["train"]["annotation_name"]
    
    def retrieve(self, sample_path):
        search_id = sample_path.split("/")[-1]
        #val_sacd
        #val_flms_fcdb
        if self.set == "train":
            index = self.ids.index(os.path.join(self.ava_root, search_id[0:-6] + ".png"))
            example = self.dataset["train"].select([index])
            retrieved_names = example["retrieved_names"][0]
            # print("sample_path:", sample_path)
            # print("original_path:","/home/zhangke/images4AVA/images/train/"+search_id[0:-6]+".png")
            # print(retrieved_names)
            retrieved_indexes = [self.ids.index(name) for name in retrieved_names]
            retrieved_embeddings= [np.asarray(self.dataset["train"].select([index])["sam_embeddings"][0]).reshape((1,64, 256)) for index in retrieved_indexes]
            return np.concatenate(retrieved_embeddings), np.asarray(example["sam_embeddings"][0]).reshape((64, 256))
        
            # index = self.synthetic_ids.index(search_id.replace(".jpg",".txt"))
            # example = self.synthetic_dataset["train"].select([index])
            # retrieved_names = example["retrieved_names"][0]
            # print("sample_path:", sample_path)
            # print("original_path:","/home/zhangke/images4AVA/images/train/"+search_id[0:-6]+".png")
            # print(retrieved_names)
            # retrieved_indexes = [self.ids.index(name) for name in retrieved_names if name.split("/")[-1].replace(".png","") != search_id[0:-6]]
            # retrieved_embeddings= [np.asarray(self.dataset["train"].select([index])["sam_embeddings"][0]).reshape((1, 64, 256)) for index in retrieved_indexes]
            # return np.concatenate(retrieved_embeddings), np.asarray(example["sam_embeddings"][0]).reshape((64, 256))
        
        if self.set == "val_flms_fcdb":
            # print(search_id, self.flms_fcdb_ids)
            index = self.flms_fcdb_ids.index(search_id.replace(".jpg",".txt"))
            example = self.flms_fcdb_dataset["train"].select([index])
            retrieved_names = example["retrieved_names"][0][0:10]
            retrieved_indexes = [self.ids.index(name) for name in retrieved_names]
            retrieved_embeddings= [np.asarray(self.dataset["train"].select([index])["sam_embeddings"][0]).reshape((1, 64, 256)) for index in retrieved_indexes]
            return np.concatenate(retrieved_embeddings), np.asarray(example["sam_embeddings"][0]).reshape((64, 256))
        
        if self.set == "val_sacd":
            index = self.sacd_ids.index(search_id.replace(".jpg",".txt"))
            example = self.sacd_dataset["train"].select([index])
            retrieved_names = example["retrieved_names"][0][0:10]
            retrieved_indexes = [self.ids.index(name) for name in retrieved_names]
            retrieved_embeddings= [np.asarray(self.dataset["train"].select([index])["sam_embeddings"][0]).reshape((1, 64, 256)) for index in retrieved_indexes]
            return np.concatenate(retrieved_embeddings), np.asarray(example["sam_embeddings"][0]).reshape((64, 256))
        
        
        # print(self.dataset["train"][0].keys())
        # row = self.df[self.df["id"]==search_id]
        # file_names = row["retrieved_names"].to_list()[0]
        # score = np.asarray([i for i in range(10)]).reshape((1,10))
        # votes = (np.asarray(row["retrieved_votes"].to_list()).reshape((100,10))*score).sum(1)
        # indices = list(np.where(votes >= threshold))[0]
        # return [file_names[indice] for indice in indices], row["embedding"]

    
    def align(self, image_retrieves):
        shape_x = max([image_retrieve.shape[1] for image_retrieve in image_retrieves])
        shape_y = max([image_retrieve.shape[2] for image_retrieve in image_retrieves])
        for i in range(len(image_retrieves)):
            image_retrieve = image_retrieves[i]
            new_image_retrieve = np.zeros((3, shape_x, shape_y))
            new_image_retrieve[:,0:image_retrieve.shape[1],0:image_retrieve.shape[2]] = image_retrieve
            image_retrieves[i] = torch.as_tensor(new_image_retrieve).unsqueeze(0)
        return torch.cat(image_retrieves, 0)

    def __getitem__(self, idx):

        image = cv2.imread(self._imgpath[idx])
        with open(self._annopath[idx], 'r') as fid:
            annotations_txt = fid.readlines()
        image, target = prepare_original_data(image, annotations_txt, self.good_thresh, self.good_num, self.nms_thresh, self.set)
        target['image_id'] = torch.as_tensor(int(self._imgpath[idx].split('/')[-1].split('.')[0]))

        if self.transform is not None:
            image, target = self.transform(copy.deepcopy(image), copy.deepcopy(target))
            # image2, target2 = self.transform(copy.deepcopy(image), copy.deepcopy(target), specify_size=image1.shape[1:3])
            
        retrieved_embeddings, embedding = self.retrieve(self._imgpath[idx])
        # print(retrieved_embeddings.shape, embedding.shape)
        # print(retrieved_embeddings.shape)
        # #retrieve:
        # target["ratios_retrieve"] = []
        # # retrieved_paths = retrieve(self._imgpath[idx], k = 1)[0:16]
        # retrieved_paths, embedding = self.retrieve(self._imgpath[idx], threshold=4.5)
        # embedding = np.asarray(embedding.to_list()[0]).reshape(64, 256)
        # retrieved_paths = retrieved_paths[0:4]
        # # print(retrieved_paths)
        # # if len(retrieved_paths) == 0:
        # #     retrieved_paths = self.retrieve(self._imgpath[idx], threshold=4.5)[0:16]
        # image_retrieves = []
        # for retrieved_path in retrieved_paths:
        #     image_retrieve = cv2.imread(retrieved_path) 
        #     ratio = (image_retrieve.shape[2]/image.shape[2], image_retrieve.shape[1]/image.shape[1])
        #     target["ratios_retrieve"].append(torch.as_tensor(ratio).unsqueeze(0))
        #     if self.transform is not None:
        #         image_retrieve, _ = self.transform(image_retrieve, None) 
        #     image_retrieves.append(image_retrieve) 
                      
        # target["ratios_retrieve"] = torch.cat(target["ratios_retrieve"],dim=0)
        # image_retrieves = self.align(image_retrieves)
        
        # return image, target, image_retrieves, embedding
        # return image, target, None, None
        return image, target, None, retrieved_embeddings, embedding

    def __len__(self):
        return len(self._imgpath)



def generate_bboxes(image):
    bins = 12.0
    h = image.shape[0]
    w = image.shape[1]
    step_h = h / bins
    step_w = w / bins
    annotations = list()
    for x1 in range(0,4):
        for y1 in range(0,4):
            for x2 in range(8,12):
                for y2 in range(8,12):
                    if (x2-x1)*(y2-y1)>0.4999*bins*bins and (y2-y1)*step_w/(x2-x1)/step_h>0.5 and (y2-y1)*step_w/(x2-x1)/step_h<2.0:
                        annotations.append([float(step_h*(0.5+x1)),float(step_w*(0.5+y1)),float(step_h*(0.5+x2)),float(step_w*(0.5+y2))])

    return annotations

def generate_bboxes_16_9(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 9
    w_step = 16
    annotations = list()
    for i in range(14,30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0,w-out_w,w_step):
                for h_start in range(0,h-out_h,h_step):
                    annotations.append([float(h_start), float(w_start), float(h_start+out_h-1), float(w_start+out_w-1)])
    return annotations

def generate_bboxes_4_3(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 16
    annotations = list()
    for i in range(14, 30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0, w-out_w,w_step):
                for h_start in range(0, h-out_h,h_step):
                    annotations.append([float(h_start),float(w_start),float(h_start+out_h-1),float(w_start+out_w-1)])
    return annotations

def generate_bboxes_1_1(image):

    h = image.shape[0]
    w = image.shape[1]
    h_step = 12
    w_step = 12
    annotations = list()
    for i in range(14,30):
        out_h = h_step*i
        out_w = w_step*i
        if out_h < h and out_w < w and out_h*out_w>0.4*h*w:
            for w_start in range(0,w-out_w,w_step):
                for h_start in range(0,h-out_h,h_step):
                    annotations.append([float(h_start),float(w_start),float(h_start+out_h-1),float(w_start+out_w-1)])
    return annotations



def build_gaic(image_set, args):
    # if image_set == "train_evaluate":
    #     imgdir = os.path.join(args.dataset_root, 'images/' + "train")
    #     annodir = os.path.join(args.dataset_root, 'annotations/' + "train")
    # else:        
    #     imgdir = os.path.join(args.dataset_root, 'images/' + image_set)
    #     annodir = os.path.join(args.dataset_root, 'annotations/' + image_set)
    if image_set == "train" or image_set == "train_evaluate":
        imgdir = os.path.join(args.dataset_root, 'images/')
        annodir = os.path.join(args.dataset_root, 'annotations/')
        
    if image_set == "val_sacd":
        imgdir = os.path.join(args.sacd_root, "images")
        annodir = os.path.join(args.sacd_root, "annotations")

    if image_set == "val_flms_fcdb":
        imgdir = os.path.join(args.fcdb_root, "images")
        annodir = os.path.join(args.fcdb_root, "annotations")

    imglist = os.listdir(imgdir)
    imgpath = []
    annopath = []
    for image in imglist:
        txt_name = image[0:-6] + '.txt'
        if image_set == "train":
            # if image in synthetic_images:
            # if image[0:-6] in list_1w:
            # if txt_name in human_list:
                # for i in range(3):
                #     imgpath.append(os.path.join(imgdir, image))
                #     annopath.append(os.path.join(annodir,  image[0:-3]+"txt"))
                # else:
                    # if random.random()<0.2:
            imgpath.append(os.path.join(imgdir, image))
            annopath.append(os.path.join(annodir,  image[0:-3]+"txt"))
        else:
            imgpath.append(os.path.join(imgdir, image))
            annopath.append(os.path.join(annodir,  image[0:-3]+"txt"))
        # imgpath.append(os.path.join(imgdir, image))
        # annopath.append(os.path.join(annodir,  image[0:-3]+"txt"))
    
    

    if image_set == "train_evaluate":
        transform = croptransform(set="val", imgsize_test=args.image_size_test)
        dataset = GAICD(args=args, set="train", imgpath=imgpath, annopath=annopath, transform=transform)
    else:
        transform = croptransform(set=image_set.split("_")[0], imgsize_test=args.image_size_test)
        dataset = GAICD(args=args, set=image_set, imgpath=imgpath,
                                annopath=annopath, transform=transform)
        
    return dataset, imgpath

