import warnings
import h5py
import os
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import Dataset
from tqdm import tqdm
import pyloudnorm as pyln


warnings.filterwarnings("ignore")


class MusicGenres(Dataset):
    GENRES = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    LABELS = {g: i for i, g in enumerate(GENRES)}
    ID_TO_LABEL = {v: k for k, v in LABELS.items()}
    BASE_DIR = "./GTZAN/Data/genres_original"
    SAMPLE_RATE = 16000
    TIME_FRAMES = 128  # About 2.1 seconds
    MEL_BANDS = 128
    MEL_FFT = 1024
    MEL_HOP_LENGTH = MEL_FFT // 4

    mels = None
    tags = None

    def __init__(self, rebuild):
        if rebuild:
            mels, tags = self.make_training_data()
            self.save(mels, tags)
        self.load()

    def __getitem__(self, index):
        mel = self.mels[index]
        mel = np.reshape(mel, (1, mel.shape[0], mel.shape[1]))
        mel = torch.from_numpy(mel)
        tag = torch.from_numpy(self.tags[index])

        return (mel, tag)

    def __len__(self):
        return self.mels.shape[0]

    def save(self, mels, tags):
        with h5py.File("./dataset.h5", "w") as outfile:
            outfile.create_dataset("mels", data=mels, dtype=np.float32)
            outfile.create_dataset("tags", data=tags, dtype=np.float32)

    def load(self):
        with h5py.File("./dataset.h5", "r") as infile:
            self.mels = infile["mels"][:]
            self.tags = infile["tags"][:]

    @classmethod
    def make_mel(cls, filepath):
        waveform, sample_rate = torchaudio.load(filepath)

        # consistent sample rate
        waveform = T.Resample(sample_rate, cls.SAMPLE_RATE)(waveform)
        sample_rate = cls.SAMPLE_RATE

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Normalize volume
        waveform = waveform.squeeze(0).numpy()
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(waveform)
        waveform = pyln.normalize.loudness(waveform, loudness, -12.0)

        # Create mel spectrogram
        waveform = torch.Tensor(waveform).view(-1)
        mel_spectrogram = T.MelSpectrogram(
            sample_rate,
            n_mels=cls.MEL_BANDS,
            n_fft=cls.MEL_FFT,
            hop_length=cls.MEL_HOP_LENGTH,
        )(waveform)
        mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
        return mel_spectrogram

    @classmethod
    def split_to_chunks(cls, mel):
        _, total_frames = mel.shape
        mel = mel[:, : -1 * (total_frames % cls.TIME_FRAMES)]

        num_chunks = total_frames // cls.TIME_FRAMES
        mel_chunks = np.split(mel, num_chunks, axis=1)
        return mel_chunks

    def make_training_data(self):
        mels = []
        tags = []
        for genre in tqdm(self.LABELS):
            dirpath = os.path.join(self.BASE_DIR, genre)
            y = np.eye(len(self.GENRES), dtype=np.float32)[
                self.LABELS[genre]
            ]  # one-hot array
            for f in tqdm(os.listdir(dirpath), leave=False):
                try:
                    filepath = os.path.join(dirpath, f)
                    mel_spectrogram = self.make_mel(filepath)
                    mel_chunks = self.split_to_chunks(mel_spectrogram)

                    mels.extend(mel_chunks)
                    tags.extend([y] * len(mel_chunks))

                except Exception as e:
                    print(f, e)
                    pass
        return mels, tags
