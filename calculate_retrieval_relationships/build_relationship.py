import numpy as np
import datasets as ds
import os
import fsspec
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import umap
import pickle
from pandas import DataFrame as df
import time
# import elasticsearch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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

def to_rgb(batch):
    batch["image"] = [img.convert("RGB") for img in batch["image"]]
    return batch

def search_samples():
    ava_dir = "/home/kzhang99/AVA_sample/cgl/"
    ava_files = find_data_files(
                ava_dir,
                splits=[
                    "train"
                    # "test"
                    # "val",
                    # "with_no_annotation",
                ],
            )
    
    ava_dataset = ds.load_dataset("parquet", data_files=ava_files)
    id_dic= dict(zip(ava_dataset["train"]["annotation_name"],ava_dataset["train"]["id"]))
    torch.save(id_dic, "/home/kzhang99/AVA_sample/id.pt")

search_samples()
   