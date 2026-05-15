import os
import torch


@torch.no_grad()
def retrieve(sample_path, img_dir, retrieval_cache_dir, split="train", k=1):
    """
    Retrieve top-k similar professional images for a given query image.

    Args:
        sample_path: path to the query image file
        img_dir: directory containing the database images
        retrieval_cache_dir: directory with precomputed .pt retrieval tables
            (cgl_{split}_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt
             and cgl_dreamsim_cache_table_paired.pt)
        split: dataset split ("train", "val", "test")
        k: number of retrieved images to return
    Returns:
        list of paths to retrieved images
    """
    img_name = os.path.basename(sample_path)
    indexes_path = os.path.join(
        retrieval_cache_dir,
        f"cgl_{split}_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt"
    )
    table_indexes = torch.load(indexes_path)
    name2id = torch.load(os.path.join(retrieval_cache_dir, "cgl_dreamsim_cache_table_paired.pt"))
    id2name = {v: k for k, v in name2id.items()}
    matched_names = [os.path.join(img_dir, id2name[i]) for i in table_indexes[img_name][:k]]
    return matched_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--retrieval_cache_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    print(retrieve(args.sample_path, args.img_dir, args.retrieval_cache_dir, args.split))