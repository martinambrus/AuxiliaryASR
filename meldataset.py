#coding: utf-8

import os
import os.path as osp
import time
import math
import random
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Sampler
from torchaudio import transforms as T

from nltk.tokenize import word_tokenize
#import phonemizer
import torchaudio.functional as AF
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


class SpecAugment:
    """Simple SpecAugment implementation for log-mel spectrograms."""

    def __init__(self,
                 apply_prob=1.0,
                 frequency_masking=None,
                 time_masking=None,
                 time_warp=None,
                 **legacy_kwargs):
        self.apply_prob = float(apply_prob)

        legacy_freq_param = legacy_kwargs.pop('freq_mask_param', None)
        legacy_num_freq_masks = legacy_kwargs.pop('num_freq_masks', None)
        legacy_time_param = legacy_kwargs.pop('time_mask_param', None)
        legacy_num_time_masks = legacy_kwargs.pop('num_time_masks', None)
        legacy_time_warp_param = legacy_kwargs.pop('time_warp_param', None)

        if legacy_kwargs:
            raise TypeError(f"Unsupported SpecAugment parameters: {list(legacy_kwargs.keys())}")

        if legacy_freq_param is not None or legacy_num_freq_masks is not None:
            frequency_masking = dict(frequency_masking or {})
            frequency_masking.setdefault('enabled', True)
            if legacy_freq_param is not None:
                frequency_masking.setdefault('freq_mask_param', legacy_freq_param)
            if legacy_num_freq_masks is not None:
                frequency_masking.setdefault('num_masks', legacy_num_freq_masks)

        if legacy_time_param is not None or legacy_num_time_masks is not None:
            time_masking = dict(time_masking or {})
            time_masking.setdefault('enabled', True)
            if legacy_time_param is not None:
                time_masking.setdefault('time_mask_param', legacy_time_param)
            if legacy_num_time_masks is not None:
                time_masking.setdefault('num_masks', legacy_num_time_masks)

        if legacy_time_warp_param is not None:
            time_warp = dict(time_warp or {})
            time_warp.setdefault('enabled', True)
            time_warp.setdefault('time_warp_param', legacy_time_warp_param)

        if frequency_masking is None:
            frequency_masking = {
                'enabled': True,
                'freq_mask_param': 27,
                'num_masks': 2,
            }

        if time_masking is None:
            time_masking = {
                'enabled': True,
                'time_mask_param': 40,
                'num_masks': 2,
            }

        if time_warp is None:
            time_warp = {
                'enabled': False,
                'time_warp_param': 5,
            }

        freq_cfg = dict(frequency_masking)
        time_cfg = dict(time_masking)
        warp_cfg = dict(time_warp)

        self.freq_mask_enabled = bool(freq_cfg.get('enabled', bool(frequency_masking)))
        self.freq_mask_param = int(freq_cfg.get('freq_mask_param', freq_cfg.get('mask_param', 27)))
        self.num_freq_masks = int(freq_cfg.get('num_masks', freq_cfg.get('num_freq_masks', 2)))
        if self.freq_mask_param <= 0 or self.num_freq_masks <= 0:
            self.freq_mask_enabled = False

        self.time_mask_enabled = bool(time_cfg.get('enabled', bool(time_masking)))
        self.time_mask_param = int(time_cfg.get('time_mask_param', time_cfg.get('mask_param', 40)))
        self.num_time_masks = int(time_cfg.get('num_masks', time_cfg.get('num_time_masks', 2)))
        if self.time_mask_param <= 0 or self.num_time_masks <= 0:
            self.time_mask_enabled = False

        self.time_warp_enabled = bool(warp_cfg.get('enabled', bool(time_warp)))
        self.time_warp_param = int(warp_cfg.get('time_warp_param', warp_cfg.get('param', 5)))
        if self.time_warp_param <= 0:
            self.time_warp_enabled = False

        self._freq_mask = None
        if self.freq_mask_enabled:
            self._freq_mask = T.FrequencyMasking(freq_mask_param=max(1, self.freq_mask_param))

        self._time_mask = None
        if self.time_mask_enabled:
            self._time_mask = T.TimeMasking(time_mask_param=max(1, self.time_mask_param))

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if self.apply_prob < 1.0 and random.random() > self.apply_prob:
            return mel_tensor

        squeeze = False
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
            squeeze = True

        augmented = mel_tensor.clone()

        if self.time_warp_enabled:
            augmented = self._apply_time_warp(augmented)

        if self.freq_mask_enabled and self._freq_mask is not None:
            for _ in range(max(0, self.num_freq_masks)):
                augmented = self._freq_mask(augmented)

        if self.time_mask_enabled and self._time_mask is not None:
            for _ in range(max(0, self.num_time_masks)):
                augmented = self._time_mask(augmented)

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def _apply_time_warp(self, mel_batch: torch.Tensor) -> torch.Tensor:
        if mel_batch.dim() != 3:
            return mel_batch

        warped = mel_batch.clone()
        batch, _, _ = warped.shape
        for idx in range(batch):
            warped[idx] = self._time_warp_single(warped[idx])
        return warped

    def _time_warp_single(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() != 2:
            return spec

        num_mels, num_steps = spec.shape
        if num_steps < 2:
            return spec

        max_warp = min(self.time_warp_param, num_steps // 2)
        if max_warp < 1:
            return spec

        low = max_warp
        high = num_steps - max_warp - 1
        if low > high:
            return spec
        if low == high:
            center = low
        else:
            center = random.randint(low, high)
        min_target = max(1, center - max_warp)
        max_target = min(num_steps - 2, center + max_warp)

        if min_target >= max_target:
            return spec

        new_center = random.randint(min_target, max_target)
        if new_center == center:
            return spec

        grid = self._build_time_warp_grid(num_mels, num_steps, center, new_center, spec.device, spec.dtype)
        warped = F.grid_sample(
            spec.unsqueeze(0).unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return warped.squeeze(0).squeeze(0)

    @staticmethod
    def _build_time_warp_grid(num_mels, num_steps, center, new_center, device, dtype):
        if new_center <= 0 or new_center >= num_steps - 1:
            return SpecAugment._identity_grid(num_mels, num_steps, device, dtype)

        t_max = num_steps - 1
        if center <= 0 or center >= t_max:
            return SpecAugment._identity_grid(num_mels, num_steps, device, dtype)

        left_scale = center / new_center
        right_scale_den = (t_max - new_center)
        if right_scale_den == 0:
            return SpecAugment._identity_grid(num_mels, num_steps, device, dtype)
        right_scale = (t_max - center) / right_scale_den

        time_idx = torch.arange(num_steps, device=device, dtype=dtype)
        src_time = torch.empty_like(time_idx)

        if new_center > 0:
            left_mask = time_idx < new_center
            src_time[left_mask] = time_idx[left_mask] * left_scale
        else:
            left_mask = torch.zeros_like(time_idx, dtype=torch.bool)

        right_mask = ~left_mask
        src_time[right_mask] = center + (time_idx[right_mask] - new_center) * right_scale
        src_time = torch.clamp(src_time, 0, t_max)

        if num_steps == 1:
            grid_time = torch.zeros(num_mels, num_steps, device=device, dtype=dtype)
        else:
            grid_time = (src_time / t_max) * 2.0 - 1.0
            grid_time = grid_time.expand(num_mels, -1)

        grid_freq = torch.linspace(-1.0, 1.0, num_mels, device=device, dtype=dtype).unsqueeze(1)
        grid_freq = grid_freq.expand(-1, num_steps)

        grid = torch.stack((grid_time, grid_freq), dim=-1).unsqueeze(0)
        return grid

    @staticmethod
    def _identity_grid(num_mels, num_steps, device, dtype):
        if num_steps <= 1:
            grid_time = torch.zeros(num_mels, num_steps, device=device, dtype=dtype)
        else:
            base = torch.linspace(-1.0, 1.0, num_steps, device=device, dtype=dtype)
            grid_time = base.expand(num_mels, -1)
        grid_freq = torch.linspace(-1.0, 1.0, num_mels, device=device, dtype=dtype).unsqueeze(1)
        grid_freq = grid_freq.expand(-1, num_steps)
        grid = torch.stack((grid_time, grid_freq), dim=-1).unsqueeze(0)
        return grid


class LengthAwareBatchSampler(Sampler):
    """Batch sampler that groups items with similar lengths together."""

    def __init__(self,
                 lengths,
                 batch_size,
                 bucket_size=None,
                 shuffle_batches=True,
                 shuffle_within_bucket=True,
                 drop_last=False,
                 seed=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.lengths = list(lengths)
        if len(self.lengths) == 0:
            raise ValueError("lengths must not be empty")

        self.batch_size = int(batch_size)
        if bucket_size is None:
            bucket_size = self.batch_size * 50
        self.bucket_size = max(self.batch_size, int(bucket_size))
        self.shuffle_batches = bool(shuffle_batches)
        self.shuffle_within_bucket = bool(shuffle_within_bucket)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        if self.seed is not None:
            rng = random.Random(self.seed + self._epoch)
        else:
            rng = random.Random()
        self._epoch += 1

        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx])

        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle_batches:
            rng.shuffle(buckets)

        batches = []
        for bucket in buckets:
            if self.shuffle_within_bucket:
                rng.shuffle(bucket)

            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle_batches:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


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
                 },
                 spec_augment_params=None,
                 validation=False
                ):

        self.data_list = data_list
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr

        mel_opts = {**{'sample_rate': sr}, **mel_params, **spect_params}
        print("Options for MEL spectrogram calculations:", mel_opts)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**mel_opts)

        self.spec_augment = None
        if spec_augment_params and not validation:
            try:
                self.spec_augment = SpecAugment(**spec_augment_params)
            except TypeError:
                logger.warning(f"Invalid SpecAugment configuration: {spec_augment_params}. Skipping augmentation.")
                self.spec_augment = None

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

            log_mel_tensor = torch.log(1e-5 + mel_tensor)

            if self.spec_augment is not None:
                log_mel_tensor = self.spec_augment(log_mel_tensor)

            acoustic_feature = (log_mel_tensor - self.mean)/self.std

            length_feature = acoustic_feature.size(1)
            acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

            return wave_tensor, acoustic_feature, text_tensor, speaker_id
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

    def __init__(self, return_wave=False, return_speaker_ids=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.return_speaker_ids = return_speaker_ids

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
        speaker_ids = torch.zeros(batch_size).long()
        for bid, (_, mel, text, speaker_id) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            speaker_ids[bid] = int(speaker_id)
            assert(text_size < (mel_size//2))

        outputs = [texts, input_lengths, mels, output_lengths]

        if self.return_speaker_ids:
            outputs.append(speaker_ids)

        if self.return_wave:
            waves = [b[0] for b in batch]
            outputs.append(waves)

        return tuple(outputs)



def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     lengths=None,
                     bucket_sampler_config=None):

    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    use_bucket_sampler = False
    batch_sampler = None

    if (not validation) and bucket_sampler_config and lengths is not None:
        lengths = list(lengths)
        if len(lengths) != len(dataset):
            raise ValueError("lengths must have the same length as path_list when using a bucket sampler")

        enabled = bucket_sampler_config.get('enabled', True)
        if enabled:
            bucket_size = bucket_sampler_config.get('bucket_size')
            if bucket_size is not None:
                bucket_size = int(bucket_size)
            else:
                multiplier = int(bucket_sampler_config.get('bucket_size_multiplier', 50))
                bucket_size = batch_size * max(1, multiplier)

            shuffle_batches = bucket_sampler_config.get('shuffle_batches', True)
            shuffle_within_bucket = bucket_sampler_config.get('shuffle_within_bucket', True)
            drop_last = bucket_sampler_config.get('drop_last', not validation)
            seed = bucket_sampler_config.get('seed')

            batch_sampler = LengthAwareBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_size=bucket_size,
                shuffle_batches=shuffle_batches,
                shuffle_within_bucket=shuffle_within_bucket,
                drop_last=drop_last,
                seed=seed,
            )
            use_bucket_sampler = True

    if use_bucket_sampler and batch_sampler is not None:
        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
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
