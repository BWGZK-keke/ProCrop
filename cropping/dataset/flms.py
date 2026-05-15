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
from scipy.io import loadmat
import fsspec
import datasets as ds
import pandas as pd

import warnings
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

def prepare_original_data(image, annos, ann_name, img_name):
    # image: image read from cv2, therefore, it is numpy array
    # annofile: 10 good boxes, may have invalid box

    good_bbox = list()

    for annotation in annos:
        # print("flms:",annotation, image.shape)
        # xmin = float(annotation[1])
        # ymin = float(annotation[0])
        # xmax = float(annotation[3])
        # ymax = float(annotation[2])
        # xmin = float(annotation[0])
        # ymin = float(annotation[1])
        # xmax = float(annotation[2])
        # ymax = float(annotation[3])
        xmin = float(annotation[0])
        ymin = float(annotation[1])
        xmax = min(float(annotation[2]),image.shape[0])
        ymax = min(float(annotation[3]),image.shape[1])
        # if xmax > image.shape[0] or ymax > image.shape[1]:
        #     print("FLMS:",xmax,ymax, image.shape)
        #     print("ann name:",ann_name)
        #     print("img name:",img_name)
        if xmin != -1:
            good_bbox.append([xmin, ymin, xmax, ymax])

    good_bbox = torch.tensor(good_bbox)

    orig_size = torch.as_tensor(list(image.shape[:2]))

    target = {'good_boxes': good_bbox,
              'orig_size': orig_size}

    return image, target



class flms(data.Dataset):

    def __init__(self, args, imgpath=None, annopath=None, anchorfile=None, transform=None):


        self._imgpath = imgpath
        self._annopath = annopath
        # self.achor_normalized = self.anchor_process(anchorfile)

        self.transform = transform
        self.good_num = args.good_num
        self.nms_thresh = args.nms_thresh
        db_root = args.retrieval_db_root
        ava_files = find_data_files(
            os.path.join(db_root, "ava_self_correlated"),
            splits=["train"],
        )
        flms_files = find_data_files(
            os.path.join(db_root, "ava_flms_fcdb"),
            splits=["train"],
        )
        self.dataset = ds.load_dataset("parquet", data_files=ava_files)
        self.flms_dataset = ds.load_dataset("parquet", data_files=flms_files)
        self.ids = self.dataset["train"]["id"]
        self.flms_ids = self.flms_dataset["train"]["annotation_name"]
    
    def anchor_process(self, anchorfile):

        anchor_normalized = []
        for acs in anchorfile:
            anchor = acs[:-1].split(' ')
            anchor = [float(ac) for ac in anchor]
            anchor_normalized.append(anchor)

        return anchor_normalized

    
    def retrieve(self, sample_path):
        search_id = sample_path.split("/")[-1]
        index = self.flms_ids.index(search_id.replace(".jpg",".txt"))
        example = self.flms_dataset["train"].select([index])
        retrieved_names = example["retrieved_names"][0]
        retrieved_indexes = [self.ids.index(name) for name in retrieved_names]
        retrieved_embeddings= [np.asarray(self.dataset["train"].select([index])["sam_embeddings"][0]).reshape((1, 64, 256)) for index in retrieved_indexes]
        return np.concatenate(retrieved_embeddings), np.asarray(example["sam_embeddings"][0]).reshape((64, 256))

    def __getitem__(self, idx):

        image = cv2.imread(self._imgpath[idx])
        annos = self._annopath[idx]
        

        retrieved_embeddings, embedding = self.retrieve(self._imgpath[idx])

        image, target = prepare_original_data(image, annos, self._annopath[idx], self._imgpath[idx])
        target['image_id'] = torch.tensor(idx)
 
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target, None, retrieved_embeddings, embedding

    def __len__(self):
        return len(self._imgpath)



def build_flms(image_set, args):

    path = os.path.join(args.flms_root, '500_image_dataset.mat')
    ann = loadmat(path)

    imgpath = []
    annopath = []

    for idx in range(500):
        name = ann['img_gt'][idx][0][0].tolist()[0]
        imgpath.append(os.path.join(args.flms_root, 'image', name))
        annopath.append(ann['img_gt'][idx][0][1].tolist())

    transform = croptransform(set='val', imgsize_test=args.image_size_test)

    dataset = flms(args=args, imgpath=imgpath,
                   annopath=annopath, anchorfile=None, transform=transform)

    return dataset, imgpath

