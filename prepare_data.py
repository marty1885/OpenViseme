import glob
import re
import os
import random
import librosa
import numpy as np
import sklearn
from sklearn.utils import shuffle

# Iterate and get the entire list of source data

wav_files = []
label_files = []

for _, _, files in os.walk('DataSet/label/'):
    for filename in files:
        fname = os.path.splitext(filename)[0]
        wav_path = f'DataSet/wav_data/{fname}.wav'
        if not os.path.exists(wav_path): continue

        wav_files += [wav_path]
        label_files += [f'DataSet/label/{fname}.txt']

assert(len(wav_files) == len(label_files))
print(f"Dataset size {len(wav_files)} files")

# Enable this block of code to only use a subset of the dataset
#subset = 1
#wav_files = wav_files[:subset]
#label_files = label_files[:subset]

# DO NOT change. Parameters are fixed by the dataset
sample_rate = 16000
shift_duration = 0.008 # 8ms
frame_duration = 0.016 # 16ms


def wav2frames(signal, sr, shift_duration, frame_duration):
    shift_length = int(sr * shift_duration)
    frame_length = int(sr * frame_duration)
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / shift_length))
    # Faster way to split signal into frames compared to for loop
    view = np.lib.stride_tricks.as_strided(signal, shape=(num_frames, frame_length), strides=(signal.strides[0]*shift_length, signal.strides[0])).copy()
    assert(len(view) == num_frames)
    return view
    """
    res = np.zeros((num_frames, frame_length))
    for frame_idx in range(num_frames):
        idx = frame_idx * shift_length
        buf = np.zeros(frame_length)
        part = signal[idx:idx+frame_length]
        buf[:len(part)] = part
        res[frame_idx] = buf
    return res
    """

label_data = [None] * len(wav_files)
loaded_frames = [None] * len(wav_files)
for i, (wav, label) in enumerate(zip(wav_files, label_files)):
    print(f"\33[2K\rProcessing {i}/{len(wav_files)} ({i/len(wav_files)*100.0}%)", end="")
    signal, _ = librosa.load(wav, sr=sample_rate)
    frames = wav2frames(signal, sample_rate, shift_duration, frame_duration)

    with open(label, 'r') as f:
        # HACK Some dataset seems to have an extra label at the end. Drop it
        num_labels = 0
        lbl = []
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            arr = [float(x) for x in line.split(',')]
            num_labels += 1
            if num_labels > len(frames):
                lbl = lbl[:-1]
            lbl += [arr]
        label_data[i] = lbl

    loaded_frames[i] = frames
wav_data = np.concatenate(loaded_frames, axis=0)
label_data = np.concatenate(label_data, axis=0)
print()
print(f"Total frames {len(wav_data)}")
print(f"Total labels {len(label_data)}")
del loaded_frames

assert(len(wav_data) == len(label_data))

good_frames = np.max(np.abs(wav_data), axis=1) > 1e-6 # Filter out frames with 0 data (quiet frames)
good_frames = good_frames & (np.max(label_data, axis=1) > 0.75) # Skip frames with low confidence
wav_data = wav_data[good_frames]
label_data = np.array(label_data)[good_frames]

del good_frames

wav = wav_data
label = [np.argmax(x) for x in label_data]

print(f"Total frames after filtering {len(wav_data)}")
print(f"Total labels after filtering {len(label_data)}")
print(f"Labels avaliable {label_data.shape[1]}")

# resample dataset to fix label bias, we take the geometric mean of the label sizes to reduce unbalance
data_dict = { k: [] for k in set(label) }
for lbl, wave in zip(label, wav):
    data_dict[lbl] += [wave]
for lbl, waves in data_dict.items():
    print(f"Label {lbl} size {len(waves)}")

def geo_mean(iterable):
    a = np.array(iterable)
    return np.exp(np.mean(np.log(a)))

label_sizes = sorted([len(x) for x in data_dict.values()])
del label_sizes[-1] # Remove the largest label (which is silence, way too many of them)
resample_size = int(geo_mean(label_sizes))
print(f"Resample data at {resample_size} per label")
for lbl, wave in data_dict.items():
    random.shuffle(wave)
    data_dict[lbl] = wave[:min(resample_size, len(wave))]

wav = []
label = []
for lbl, wave in data_dict.items():
    wav += wave
    label += [lbl] * len(wave)
wav = np.array(wav)
label = np.array(label)
print(f"Final dataset size {len(wav)}")

del wav_data

# Convert into MFCC
#windowing = np.hanning(wav.shape[1])
#wav = wav * windowing
mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=24, n_fft=256, hop_length=128, fmin=120, fmax=4000, center=False)
mfcc = mfcc.reshape((mfcc.shape[0], -1))

# normalize mfcc
std = np.mean(np.std(mfcc, axis=1))
mean = np.mean(mfcc)
print(f"MFCC std {std}")
print(f"MFCC mean {mean}")
mfcc = (mfcc - mean) / std

# Shuffle the dataset
mfcc, label = shuffle(mfcc, label)

np.save('DataSet/mel.npy', mfcc)
np.save('DataSet/label.npy', label)
