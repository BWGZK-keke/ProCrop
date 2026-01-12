import numpy as np
import datasets as ds
import os
import fsspec
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
    gaic_data_dir = "/home/kzhang99/GAICv2_dataset/embedding_db/"
    ava_dir = "/home/kzhang99/GAICv2_train_dataset/embedding_db/"
    ava_files = find_data_files(
                ava_dir,
                splits=[
                    "train"
                ],
            )
    
    gaic_files = find_data_files(
                gaic_data_dir,
                splits=[
                    "train",
                    # "test",
                    # "val",
                ],
            )
    # print(data_files)
    ava_dataset = ds.load_dataset("parquet", data_files=ava_files)
    gaic_dataset = ds.load_dataset("parquet", data_files=gaic_files)

    ava_dataset["train"].load_faiss_index('embeddings', '/home/kzhang99/RALF_code/cache/GAICv2_train_full/cgl_dreamsim_wo_head_index.faiss')
    ids = ava_dataset["train"]["id"]
    # ava_dataset["train"] = ava_dataset["train"].select(range(55000))
    # gaic_dataset["train"] = gaic_dataset["train"].select(range(10))
    for key in ["train"]:
        print("start search")
        time_start = time.time()
        retrieved_names = []
        retrieved_votes = []
        count = 0
        count_time = 0
        # print(len(gaic_dataset[key]))
        for i in range(len(gaic_dataset[key])):  
            if i % 100 == 0:
                print(i)    
            embedding_target = np.array(gaic_dataset[key][i]["sam_embeddings"])
            if i > 2:
                start_time=time.time()
            # print(ava_dataset["train"][0])
            scores, retrieved_examples = ava_dataset["train"].get_nearest_examples('embeddings',embedding_target, k=11)
            retrieved_name = retrieved_examples["annotation_name"]
            # if i > 2:
            #     end_time = time.time()
            #     count +=1
            # if i > 2:
            #     count_time = count_time+ end_time-start_time
            #     print(end_time-start_time, count_time/count)
            # print(retrieved_name)
            vote_keys = [i for i in list(retrieved_examples.keys()) if "vote" in i]
            retrieved_vote = []
            for vote_key in vote_keys:
                retrieved_vote.append(retrieved_examples[vote_key])
            retrieved_vote = np.array(retrieved_vote).transpose()
            retrieved_votes.append(retrieved_vote.flatten())
            retrieved_names.append(retrieved_name)

        # gaic_dataset[key] =  gaic_dataset[key].add_column("retrieved_image", retrieved_datas)
        gaic_dataset[key] =  gaic_dataset[key].add_column("retrieved_names", retrieved_names)
        gaic_dataset[key] =  gaic_dataset[key].add_column("retrieved_votes", retrieved_votes)



    output_dir = "/home/kzhang99/GAICv2_self_hdfs_embed_root/"

    num_shards = 2
    for index in range(num_shards):
        shard = gaic_dataset["train"].shard(num_shards, index)
        parquet_name = f"train-{index:05d}-of-{num_shards:05d}.parquet"
        shard.to_parquet(str(output_dir + parquet_name))
        print(output_dir+parquet_name)

   
os.environ['HF_HOME'] = '/data/kzhang99/.cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/kzhang99/.cache'
os.environ['TRANSFORMERS_CACHE'] = '/data/kzhang99/.cache'
os.environ['HF_DATASETS_CACHE'] = '/data/kzhang99/.cache'
search_samples()

#1.67 
#0.94
#