#coding: utf-8

import os
import os.path as osp
import time
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Sampler
from typing import Any, Dict, Optional

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


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AdaptiveMaskingAxisConfig:
    """Configuration for adaptive masking along a single spectrogram axis."""

    def __init__(self, axis_name: str, global_enabled: bool, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        self.axis_name = axis_name
        self._active = global_enabled and bool(config.get('enabled', True))
        self.ratio = _to_float(config.get('ratio'))
        self.min_ratio = _to_float(config.get('min_ratio'))
        self.max_ratio = _to_float(config.get('max_ratio'))
        self.min_masks = _to_int(config.get('min_masks'))
        self.max_masks = _to_int(config.get('max_masks'))
        self.min_width = _to_int(config.get('min_width'))
        self.max_width = _to_int(config.get('max_width'))

    def resolve_mask_count(self, base_count: int) -> int:
        if not self._active:
            return int(base_count)

        count = int(base_count)
        lower = self.min_masks if self.min_masks is not None else count
        upper = self.max_masks if self.max_masks is not None else count
        lower = max(0, lower)
        upper = max(lower, upper)
        if lower == upper:
            return lower
        return random.randint(lower, upper)

    def resolve_mask_width(self, base_width: int, axis_length: int) -> int:
        if axis_length <= 0:
            return 0

        width = int(base_width)
        if self._active:
            if self.ratio is not None:
                width = int(round(axis_length * self.ratio))
            if self.min_ratio is not None:
                width = max(width, int(math.floor(axis_length * self.min_ratio)))
            if self.max_ratio is not None:
                width = min(width, int(math.ceil(axis_length * self.max_ratio)))
            if self.min_width is not None:
                width = max(width, self.min_width)
            if self.max_width is not None:
                width = min(width, self.max_width)

        width = max(0, min(width, axis_length))
        if axis_length > 0 and width == 0:
            width = 1
        return width


class AdaptiveMaskingConfig:
    """Helper class that normalises adaptive masking configuration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = dict(config or {})
        self.enabled = bool(config.get('enabled', False))
        time_cfg = self._merge_axis_config(config, 'time')
        freq_cfg = self._merge_axis_config(config, 'freq')
        self.time = AdaptiveMaskingAxisConfig('time', self.enabled, time_cfg)
        self.freq = AdaptiveMaskingAxisConfig('freq', self.enabled, freq_cfg)

    @staticmethod
    def _merge_axis_config(config: Dict[str, Any], axis: str) -> Dict[str, Any]:
        axis_config = dict(config.get(axis, {}))

        legacy_keys = {
            'ratio': f'{axis}_mask_ratio',
            'max_ratio': f'{axis}_mask_max_ratio',
            'min_ratio': f'{axis}_mask_min_ratio',
            'min_masks': f'min_{axis}_masks',
            'max_masks': f'max_{axis}_masks',
            'min_width': f'min_{axis}_mask_width',
            'max_width': f'max_{axis}_mask_width',
        }

        for target, legacy in legacy_keys.items():
            if target not in axis_config and legacy in config:
                axis_config[target] = config[legacy]

        if 'enabled' not in axis_config and f'{axis}_enabled' in config:
            axis_config['enabled'] = config[f'{axis}_enabled']

        return axis_config


class SpecAugment:
    """SpecAugment implementation with optional adaptive masking."""

    def __init__(self,
                 freq_mask_param: int = 27,
                 time_mask_param: int = 40,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 apply_prob: float = 1.0,
                 enabled: bool = True,
                 mask_value: float = 0.0,
                 adaptive_masking: Optional[Dict[str, Any]] = None):
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.apply_prob = float(apply_prob)
        self.enabled = bool(enabled)
        self.mask_value = float(mask_value)
        self.adaptive_masking = AdaptiveMaskingConfig(adaptive_masking)

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return mel_tensor

        if self.apply_prob < 1.0 and random.random() > self.apply_prob:
            return mel_tensor

        squeeze = False
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
            squeeze = True

        augmented = mel_tensor.clone()

        freq_masks = self.adaptive_masking.freq.resolve_mask_count(self.num_freq_masks)
        time_masks = self.adaptive_masking.time.resolve_mask_count(self.num_time_masks)
        freq_width = self.adaptive_masking.freq.resolve_mask_width(self.freq_mask_param, augmented.size(-2))
        time_width = self.adaptive_masking.time.resolve_mask_width(self.time_mask_param, augmented.size(-1))

        augmented = self._apply_mask_along_axis(augmented, axis=-2, max_width=freq_width, num_masks=freq_masks)
        augmented = self._apply_mask_along_axis(augmented, axis=-1, max_width=time_width, num_masks=time_masks)

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def _apply_mask_along_axis(self, tensor: torch.Tensor, axis: int, max_width: int, num_masks: int) -> torch.Tensor:
        if num_masks <= 0 or max_width <= 0:
            return tensor

        dim = tensor.size(axis)
        if dim <= 0:
            return tensor

        max_width = max(0, min(max_width, dim))
        if max_width == 0:
            return tensor

        for _ in range(num_masks):
            width = random.randint(0, max_width)
            if width == 0:
                continue
            start = random.randint(0, dim - width)
            if axis == -1 or axis == tensor.dim() - 1:
                tensor[..., start:start + width] = self.mask_value
            elif axis == -2 or axis == tensor.dim() - 2:
                tensor[..., start:start + width, :] = self.mask_value
            else:
                # Generic masking for arbitrary axes.
                index = [slice(None)] * tensor.dim()
                index[axis] = slice(start, start + width)
                tensor[tuple(index)] = self.mask_value
        return tensor


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
                enabled = spec_augment_params.get('enabled', True)
            except AttributeError:
                enabled = True

            if enabled:
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
