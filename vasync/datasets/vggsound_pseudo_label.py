import os
import glob
from sklearn.cluster import KMeans
import numpy as np


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


def main():
    ds_path = "/raid/hhemati/Datasets/MultiModal/VGGSound/"
    metadata_path = "/netscratch/hhemati/Datasets/MultiModal/VGGSound/metadata.txt"
    num_clusters = 30

    all_features, all_video_ids= load_vgg_features(ds_path, metadata_path)
    labels = pseudo_labels(all_features, num_clusters)
    
    with open(os.path.join(ds_path, f"meta_pseudo_{num_clusters}.txt"), "w") as metafile:
        for itr, video_id in enumerate(all_video_ids):
            metafile.write(video_id + "|" + str(labels[itr]) + "\n")
    print("Finished.")


if __name__ == "__main__":
    main()