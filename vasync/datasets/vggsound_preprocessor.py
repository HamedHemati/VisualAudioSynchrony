from vasync.limit_threads import *
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import librosa
import shutil
import soundfile 
import scipy
import numpy as np
from scipy.io import wavfile
import cv2
import random
import torch
import glob


# ======================
# ====================== Convert Video to WAV
# ======================
def convert_mp4_to_wav(source_path, target_path):
    try:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(source_path, target_path))
        return False
    except:
        print(f"Error occured for {source_path}")
        return source_path

def convert_videos_to_wav(ds_path):
    with open(os.path.join(ds_path, "metadata.txt"), "r") as metafile:
        metadata = metafile.readlines()
    metadata = [l.strip() for l in metadata]
    videos = [l.split("|")[0] for l in metadata]
    dest_loc = os.path.join(ds_path, "wavs")
    os.makedirs(dest_loc, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=20)
    count_videos = len(videos)
    errors = []
    futures = []
    for itr, video in enumerate(videos):
        print(f"Converting video {itr}/{count_videos}.")
        source_path = os.path.join(ds_path, "videos", video + ".mp4")
        dest_path = os.path.join(dest_loc, video + ".wav")
        result = executor.submit(partial(convert_mp4_to_wav, source_path, dest_path))
        futures.append(result)
    
    metadata = [future.result() for future in futures]

    with open(os.path.join(ds_path, "errors_wavs.txt"), "w") as error_file:
        for error in errors:
            error_file.write(error + "\n")
    
    print("Finished converting mp4 files to wav.")


# ======================
# ====================== Extract Random Frames
# ======================
def get_frame(video_cap, sec):
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.set(1, int(sec * fps))
    success, image_frame = video_cap.read()
    if success:
        return image_frame
    return None


def get_frame_random(video_cap, start, offset):
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    low = int(start * fps)
    high = int((start + offset) * fps)
    max_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if high >= max_frames:
        high = max_frames - 1
    rand_fr = random.randint(low, high)
    
    video_cap.set(1, rand_fr)
    success, image_frame = video_cap.read()
    if success:
        return image_frame
    return None


def save_frame(cv2_img, out_path):
    cv2.imwrite(out_path, cv2_img)


def extract_frame(source_path, dest_path, start, length):
    video_cap = cv2.VideoCapture(source_path)
    frame = get_frame_random(video_cap, start=start, offset=length)
    if frame is not None:
        save_frame(frame, dest_path)
    else:
        print(f"Video frame for {source_path} could not be extracted.")


def extract_video_frames(ds_path, length=5):
    with open(os.path.join(ds_path, "metadata.txt"), "r") as metafile:
        metadata = metafile.readlines()
    metadata = [l.strip() for l in metadata]
    videos = [(l.split("|")[0], int(l.split("|")[2])) for l in metadata]
    dest_loc = os.path.join(ds_path, "preprocessed", "random_frames")
    os.makedirs(dest_loc, exist_ok=True)

    count_videos = len(videos)
    executor = ProcessPoolExecutor(max_workers=20)
    futures = []
    for itr, (video, start) in enumerate(videos):
        print(f"Converting video {itr}/{count_videos}.")
        source_path = os.path.join(ds_path, "videos", video + ".mp4")
        dest_path = os.path.join(dest_loc, video + ".jpg")
        result = executor.submit(partial(extract_frame, source_path, dest_path, start, length))
        futures.append(result)
    
    metadata = [future.result() for future in futures]

    print("Finished converting mp4 files to wav.")


# ======================
# ====================== Create final format DS
# ======================
def convert_and_copy(itr, count_videos, output_path, video, frame_path, wav_path, start, length, lbl, split):
    print(f"Converting {itr}/{count_videos}")
    try:
        out_path_frame = os.path.join(output_path, "images", video + ".jpg")
        out_path_wav = os.path.join(output_path, "wavs", video + ".wav")
        
        # Copy image frame
        shutil.copyfile(frame_path, out_path_frame)

        # Load wav segment 
        # wav, sr = librosa.load(wav_path, mono=True)
        # wav, sr = soundfile.read(wav_path)
        PCM16_RANGE =32768.0
        
        sr, wav = wavfile.read(wav_path)
        wav = wav / PCM16_RANGE

        low = int(start * sr)
        high = int(start * sr + length * sr)
        if high >= len(wav):
            high = len(wav) - 1
        wav_seg = wav[low:high]
        
        # Save wav segment
        wav_seg = np.asfortranarray(wav_seg[:, 0])
        librosa.output.write_wav(out_path_wav, wav_seg, sr)
        out_metaline =  f"{video}|{lbl}|{split}"
        return out_metaline

    except:
        return None
        

def create_final_ds(args, output_path, length=5):
    with open(os.path.join(ds_path, "metadata.txt"), "r") as metafile:
        metadata = metafile.readlines()
    metadata = [l.strip() for l in metadata]
    videos = [(l.split("|")[0], int(l.split("|")[2]), l.split("|")[3], l.split("|")[4]) for l in metadata]
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    count_videos = len(videos)
    executor = ProcessPoolExecutor(max_workers=20)
    futures = []
    metadata_out = []
    for itr, (video, start, lbl, split) in enumerate(videos):
        frame_path = os.path.join(ds_path, "preprocessed/random_frames", video + ".jpg")
        wav_path = os.path.join(ds_path, "wavs", video + ".wav")
        if not os.path.exists(frame_path) or not os.path.exists(wav_path):
            print(f"Error when copying {video}")
            pass
        result = executor.submit(partial(convert_and_copy, itr, count_videos, output_path, video, frame_path, wav_path, start, length, lbl, split))
        futures.append(result)

    metadata_out = [future.result() for future in futures]
    metadata_out = [l for l in metadata_out if l is not None]
    with open(os.path.join(output_path, "metadata.txt"), "w") as metafile:
        for l in metadata_out:
            metafile.write(l + "\n")

    print("Finished successfully.")


# ======================
# ====================== Extract audio features
# ======================
def extract_audio_features_vggish(ds_path):
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    model.eval()
    features_path = os.path.join(ds_path, "audio_features_vggish")
    os.makedirs(features_path, exist_ok=True)

    list_wavs = glob.glob(os.path.join(ds_path, "wavs", "*.wav"))   
    for itr, wav_path in enumerate(list_wavs):
        print(f"Extracting {itr}/{len(list_wavs)}")
        video_id = os.path.basename(wav_path).split(".")[0]
        try:
            out = model.forward(wav_path)
            # out = torch.mean(out, dim=0).detach().cpu().numpy()
            out = out[-1].detach().cpu().numpy()
            np.save(os.path.join(features_path, video_id + ".npy"), out)
        except:
            print(f"Skipping {video_id}")
            
# ======================
# ====================== Main
# ======================
if  __name__ == "__main__":
    operation = sys.argv[1]
    ds_path = sys.argv[2]

    if operation == "convert_to_wav":
        convert_videos_to_wav(ds_path)
    elif operation == "extract_frames":
        extract_video_frames(ds_path)
    elif operation == "create_final_ds":
        output_path = sys.argv[3]
        create_final_ds(ds_path, output_path)
    elif operation == "extract_audio_features_vggish":
        extract_audio_features_vggish(ds_path)
    else:
        raise RuntimeError("Operation not defined")
