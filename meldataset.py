#coding: utf-8

import os
import os.path as osp
import time
import random
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from nltk.tokenize import word_tokenize
#import phonemizer
import torchaudio.functional as AF
import math
from typing import Iterator, List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_utils import TextCleaner
np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
# SPECT_PARAMS = {
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }
# MEL_PARAMS = {
#     "n_mels": 80,
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }

#global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dict_path=DEFAULT_DICT_PATH,
                 sr=24000,
                 spect_params={
                     "n_fft": 2048,
                     "win_length": 1200,
                     "hop_length": 300
                 },
                 mel_params={
                     "n_mels": 80
                 }
                ):

        self.data_list = data_list
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr

        # make a copy so the caller's dictionary is not modified
        mel_params = dict(mel_params) if mel_params is not None else {}

        # Ensure that the mel spectrogram configuration matches the audio
        # processing configuration.  In the default configuration this keeps the
        # hop length and FFT size consistent with the resampled waveform.  Using
        # mismatched parameters previously led to unstable behaviour,
        # particularly on shorter datasets where every frame matters.
        mel_params.setdefault("sample_rate", sr)
        if spect_params is not None:
            for key in ("n_fft", "win_length", "hop_length"):
                if key in spect_params and key not in mel_params:
                    mel_params[key] = spect_params[key]

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**mel_params)

        # https://github.com/yl4579/StyleTTS/issues/57
        self.mean, self.std = -4, 4

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]
            wave, text_tensor, speaker_id = self._load_tensor(data)
            wave_tensor = torch.from_numpy(wave).float()
            mel_tensor = self.to_melspec(wave_tensor)

            if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
                mel_tensor = F.interpolate(
                    mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                    mode='linear').squeeze(0)

            acoustic_feature = (torch.log(1e-5 + mel_tensor) - self.mean)/self.std

            length_feature = acoustic_feature.size(1)
            acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

            return wave_tensor, acoustic_feature, text_tensor, data[0]
        except Exception as e:
            try:
                wave_path, text, speaker_id = data
                print(f"Error for wave path: {wave_path}, skipping - {e}")
            except Exception as e2:
                print(f"Error for wave data: {data}, skipping - {e2}")

            # Fallback to another index to keep training going
            new_idx = (idx + 1) % len(self.data_list)
            return self.__getitem__(new_idx)

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id) if speaker_id else 0

        wave_tensor, sr = torchaudio.load(wave_path)

        if wave_tensor.size(0) > 1:
            wave_tensor = wave_tensor.mean(dim=0)
            print("using mono track from stereo WAV file: ", wave_path)
        else:
            wave_tensor = wave_tensor.squeeze(0)

        # convert into the correct sample rate, if not correct yet
        if sr != self.sr:
            wave_tensor = AF.resample(wave_tensor, sr, self.sr)
            print("resampling: ", wave_path, ", from: ", sr, ", to: ", self.sr)
        wave = wave_tensor.numpy()

        # # phonemize the text
        # ps = self.g2p(text.replace('-', ' '))
        # if "'" in ps:
        #     ps.remove("'")

        # if wave.shape[-1] == 2:
        #     wave = wave[:, 0].squeeze()
        # if sr != 24000:
        #     wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        #     print(wave_path, sr)

        #ps = global_phonemizer.phonemize([text])
        #print(ps[0])
        # print("-----------phonemized------------")
        #ps = word_tokenize(ps[0])
        #print(text)
        ps = word_tokenize(text)
        #print(ps)
        ps = ' '.join(ps)
        #print(ps)
          # phonemize the text
        ps = ps.replace("(", "-")
        ps = ps.replace(")", "-")


        text = self.text_cleaner(ps)
        #print(text)
        #exit()
        # print("-----------cleaned----------------")
        blank_index = self.text_cleaner.word_index_dictionary[" "]
        text.insert(0, blank_index) # add a blank at the beginning (silence)
        text.append(blank_index) # add a blank at the end (silence)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id




class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]
        for bid, (_, mel, text, path) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert(text_size < (mel_size//2))

        if self.return_wave:
            waves = [b[0] for b in batch]
            return texts, input_lengths, mels, output_lengths, paths, waves

        return texts, input_lengths, mels, output_lengths


class LengthBucketBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Batch sampler that keeps similarly long items together.

    The dataset is expected to be pre-sorted by utterance duration.  During
    iteration the order of the batches (and optionally the order inside a
    batch) is shuffled which preserves the SortaGrad curriculum while still
    providing enough randomness for robust training.
    """

    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 batch_size: int,
                 drop_last: bool,
                 shuffle: bool = True,
                 seed: int = None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._random = random.Random(seed)

    def __len__(self) -> int:
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        return math.ceil(length / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]

        if self.shuffle:
            self._random.shuffle(batches)
            for batch in batches:
                self._random.shuffle(batch)

        for batch in batches:
            yield batch



def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     bucket_sampler=False,
                     bucket_sampler_shuffle=True,
                     bucket_sampler_seed=None):

    dataset = MelDataset(path_list, **dataset_config)
    collate_fn = Collater(**collate_config)
    if bucket_sampler and (not validation):
        sampler = LengthBucketBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=bucket_sampler_shuffle,
            seed=bucket_sampler_seed)
        data_loader = DataLoader(dataset,
                                 batch_sampler=sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=(not validation),
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))

    return data_loader
