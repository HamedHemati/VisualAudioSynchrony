from vasync.utils.limit_threads import *
import os
import glob
from sklearn.cluster import KMeans
import numpy as np


#####
#    Extract audio features and compute pseudo labels
#####
def load_vgg_features(ds_path, metadata_path):
    print("Loading audio features")
    features_path = os.path.join(ds_path, "audio_features_vggish")
    all_features_paths = glob.glob(os.path.join(features_path, "*.npy"))
    all_features = [np.load(p) for p in all_features_paths]
    all_video_ids = [os.path.basename(p).split(".")[0] for p in all_features_paths]
    
    # with open(metadata_path, "r") as metafile:
    #     metadata_lines = metafile.readlines()
    # video_to_cls = {l.strip().split("|")[0]:l.strip().split("|")[3] for l in metadata_lines}
    # all_classes = list(set([l for l in video_to_cls.values()]))
    # cls_to_id = {c:i for (i, c) in enumerate(all_classes)}
    # id_to_cls = {i:c for (i, c) in enumerate(all_classes)}
    # all_clnames = [video_to_cls[vid] for vid in all_video_ids]
    # all_ids = [cls_to_id[video_to_cls[vid]] for vid in all_video_ids]
    
    return all_features, all_video_ids #, all_ids, all_clnames


def pseudo_labels(features, num_clusters):
    print("Computing pseudo labels ...")
    features_kmeans = np.array(features)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(features_kmeans)
    psuedo_labels = kmeans.predict(features_kmeans)
    return psuedo_labels


def extract_pseudo_labels(ds_path, metadata_path, num_clusters):
    all_features, all_video_ids= load_vgg_features(ds_path, metadata_path)
    labels = pseudo_labels(all_features, num_clusters)
    
    with open(os.path.join(ds_path, f"meta_pseudo_{num_clusters}.txt"), "w") as metafile:
        for itr, video_id in enumerate(all_video_ids):
            metafile.write(video_id + "|" + str(labels[itr]) + "\n")
    print("Finished extracting pseudo labels.")


#####
#    Remove ids whose corresponding files are missing 
#####
def remove_missing_ids(ds_path, num_clusters):
    with open(os.path.join(ds_path, f"meta_pseudo_{num_clusters}.txt"), "r") as metafile:
        all_lines = metafile.readlines()
    all_lines = [l.strip() for l in all_lines]
    all_lines = [(l.split("|")[0], l.split("|")[1]) for l in all_lines]
    images_path = os.path.join(ds_path, "images")
    exists = [os.path.exists(os.path.join(images_path, l[0] + ".jpg")) for l in all_lines]
    
    with open(os.path.join(ds_path, f"meta_pseudo_{num_clusters}_final.txt"), "w") as metafile:
        for itr, line in enumerate(all_lines):
            if exists[itr]:
                metafile.write(line[0] + "|" + line[1] + "\n")



#####
#     Copy real labels
#####
def copy_real_labels(ds_path, original_metadata, num_clusters):
    with open(original_metadata, "r") as metafile:
        metadata_lines = metafile.readlines()
    video_to_cls = {l.strip().split("|")[0]:l.strip().split("|")[3] for l in metadata_lines}
    all_cls = [l.strip().split("|")[3].split(",")[0].strip() for l in metadata_lines]
    all_cls = set(all_cls)
    print(f"Number of classes {len(all_cls)}")
    cls_to_id = {cl:i for (i, cl) in enumerate(all_cls)}
    with open(os.path.join(ds_path, f"meta_pseudo_{num_clusters}.txt"), "r") as metafile:
        all_lines = metafile.readlines()
    all_lines = [l.strip() for l in all_lines]
    all_lines = [(l.split("|")[0], l.split("|")[1]) for l in all_lines]
    images_path = os.path.join(ds_path, "images")
    exists = [os.path.exists(os.path.join(images_path, l[0] + ".jpg")) for l in all_lines]
    with open(os.path.join(ds_path, f"meta_real_final.txt"), "w") as metafile:
        for itr, line in enumerate(all_lines):
            if exists[itr]:
                cl = video_to_cls[line[0]].split(",")[0].strip()
                metafile.write(line[0] + "|" + str(cls_to_id[cl]) +"|"+ cl + "\n")

def main():
    ds_path = "/raid/hhemati/Datasets/MultiModal/VGGSound/"
    metadata_path = "/netscratch/hhemati/Datasets/MultiModal/VGGSound/metadata.txt"
    num_clusters = 30
    
    # extract_pseudo_labels(ds_path, metadata_path, num_clusters)
    # remove_missing_ids(ds_path, num_clusters)
    copy_real_labels(ds_path, metadata_path, num_clusters)


if __name__ == "__main__":
    main()