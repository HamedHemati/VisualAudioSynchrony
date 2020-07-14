import sys
import os
from pytube import YouTube   
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def download_yt_video(video_id, video_save_path, video_name):
    YouTube(f"https://www.youtube.com/watch?v={video_id}").streams.first().download(output_path=video_save_path, filename=video_name) 


def download_video(line, itr, videos_path):
    video_id = line[0]
    video_name = f"video_{itr}"
    if type(line[2]) == list:
        labels = [lbl.replace('"', '') for lbl in line[2]]
        labels = ",".join(labels)
    else:
        labels = line[2]
        
    try:
        download_yt_video(video_id, videos_path, video_name)
    except:
        print(f"Problem accured for video {video_id}")
        return None
    return [video_name, video_id, line[1], labels, line[3]]
    

def main(ds_csv_path, output_path):
    # Create directories
    os.makedirs(output_path, exist_ok=True)
    videos_path = os.path.join(output_path, "videos")
    os.makedirs(videos_path, exist_ok=True)

    # Read csv metadata
    with open(ds_csv_path, "r") as metadata:
        all_lines = metadata.readlines()
    all_lines = [l.strip().split(",") for l in all_lines]
    # Lines with more than two labels
    all_lines = [[l[0], l[1], l[2:-1], l[-1]] if len(l)>4 else l for l in all_lines]
    
    # Download videos
    futures = []
    executor = ProcessPoolExecutor(max_workers=20)
    for itr, line in enumerate(all_lines):
        print(f"Downloading {itr}/{len(all_lines)}")
        out = executor.submit(partial(download_video, line, itr, videos_path))
        futures.append(out)
    
    metadata = [future.result() for future in futures]
    metadata = [m for m in metadata if m is not None]

    # Save the metadata
    metadata_path = os.path.join(output_path, "metadata.txt")
    with open(metadata_path, "w") as metadata_file:
        for line in metadata:
            metadata_file.write("|".join(line) + "\n")

if __name__ == "__main__":
    ds_csv_path = sys.argv[1]
    output_path = sys.argv[2]
    assert os.path.isfile(ds_csv_path)
    assert output_path != ""
    
    main(ds_csv_path, output_path)