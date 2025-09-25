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
                 freq_mask_param=27,
                 time_mask_param=40,
                 num_freq_masks=2,
                 num_time_masks=2,
                 apply_prob=1.0):
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.apply_prob = float(apply_prob)

        self._freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        self._time_mask = T.TimeMasking(time_mask_param=self.time_mask_param)

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if self.apply_prob < 1.0 and random.random() > self.apply_prob:
            return mel_tensor

        squeeze = False
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
            squeeze = True

        augmented = mel_tensor.clone()

        for _ in range(max(0, self.num_freq_masks)):
            augmented = self._freq_mask(augmented)

        for _ in range(max(0, self.num_time_masks)):
            augmented = self._time_mask(augmented)

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented


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
                 validation=False,
                 auxiliary_objectives=None
                ):

        self.data_list = data_list
        self.text_cleaner = TextCleaner(dict_path)
        self.sr = sr
        self.auxiliary_objectives = auxiliary_objectives or {}

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

        pron_cfg = self.auxiliary_objectives.get('pronunciation_error', {}) if self.auxiliary_objectives else {}
        self.pronunciation_error_enabled = bool(pron_cfg.get('enabled', False))
        self.pronunciation_error_map = {}
        if self.pronunciation_error_enabled:
            label_map_path = pron_cfg.get('label_map_path')
            if label_map_path:
                self.pronunciation_error_map = self._load_label_map(label_map_path)

        frame_cfg = self.auxiliary_objectives.get('frame_phoneme', {}) if self.auxiliary_objectives else {}
        self.frame_phoneme_enabled = bool(frame_cfg.get('enabled', False))
        self.frame_label_source = frame_cfg.get('label_source', 'uniform_alignment')
        self.frame_label_dir = frame_cfg.get('label_dir')
        self.frame_label_extension = frame_cfg.get('label_extension', '.npy')

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

            aux_targets = {}
            if self.pronunciation_error_enabled:
                aux_targets['pronunciation_error'] = self._get_pronunciation_error_label(data[0])

            if self.frame_phoneme_enabled and self.frame_label_source == 'file':
                frame_labels = self._load_frame_labels(data[0])
                if frame_labels is not None:
                    aux_targets['frame_phoneme'] = frame_labels

            return wave_tensor, acoustic_feature, text_tensor, data[0], int(speaker_id), aux_targets
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

    def _load_label_map(self, path):
        mapping = {}
        if not os.path.exists(path):
            logger.warning(f"Pronunciation error label map not found at {path}")
            return mapping
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned or cleaned.startswith('#'):
                    continue
                parts = cleaned.split('|')
                if len(parts) != 2:
                    continue
                mapping[parts[0].strip()] = int(parts[1])
        return mapping

    def _get_pronunciation_error_label(self, path):
        if not self.pronunciation_error_map:
            return 0
        return self.pronunciation_error_map.get(path, 0)

    def _load_frame_labels(self, path):
        if not self.frame_label_dir:
            return None
        base = os.path.splitext(os.path.basename(path))[0]
        target_path = os.path.join(self.frame_label_dir, base + self.frame_label_extension)
        if not os.path.exists(target_path):
            return None
        try:
            labels = np.load(target_path)
            if not isinstance(labels, np.ndarray):
                return None
            return torch.from_numpy(labels.astype(np.int64))
        except Exception as exc:
            logger.warning(f"Failed to load frame labels for {path}: {exc}")
            return None




class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False, auxiliary_objectives=None, include_auxiliary_targets=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.auxiliary_objectives = auxiliary_objectives or {}
        self.include_auxiliary_targets = include_auxiliary_targets

        frame_cfg = self.auxiliary_objectives.get('frame_phoneme', {}) if self.auxiliary_objectives else {}
        self.frame_phoneme_enabled = bool(frame_cfg.get('enabled', False))
        self.frame_label_source = frame_cfg.get('label_source', 'uniform_alignment')

        speaker_cfg = self.auxiliary_objectives.get('speaker_embedding', {}) if self.auxiliary_objectives else {}
        self.speaker_embedding_enabled = bool(speaker_cfg.get('enabled', False))

        pron_cfg = self.auxiliary_objectives.get('pronunciation_error', {}) if self.auxiliary_objectives else {}
        self.pronunciation_error_enabled = bool(pron_cfg.get('enabled', False))

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
        aux_per_sample = []
        for bid, (_, mel, text, path, speaker_id, aux_data) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            assert(text_size < (mel_size//2))
            aux_per_sample.append({
                'speaker_id': speaker_id,
                'aux_data': aux_data,
                'mel_length': mel_size,
                'text_length': text_size,
                'text': text
            })

        if self.return_wave:
            waves = [b[0] for b in batch]
            base_outputs = [texts, input_lengths, mels, output_lengths, paths, waves]
        else:
            base_outputs = [texts, input_lengths, mels, output_lengths]

        if not self.include_auxiliary_targets:
            return tuple(base_outputs)

        aux_targets = {}

        if self.speaker_embedding_enabled:
            speaker_ids = torch.tensor([sample['speaker_id'] for sample in aux_per_sample]).long()
            aux_targets['speaker_ids'] = speaker_ids

        if self.frame_phoneme_enabled:
            frame_targets = torch.full((batch_size, max_mel_length), fill_value=-100, dtype=torch.long)
            for bid, sample in enumerate(aux_per_sample):
                mel_len = sample['mel_length']
                provided = sample['aux_data'].get('frame_phoneme') if sample['aux_data'] else None
                if provided is not None:
                    frame = provided
                    if frame.dim() == 1:
                        frame = frame.long()
                    frame = frame[:mel_len]
                else:
                    frame = self._uniform_frame_targets(sample['text'], mel_len)
                frame_targets[bid, :frame.shape[0]] = frame
            aux_targets['frame_phoneme'] = frame_targets

        if self.pronunciation_error_enabled:
            pron_labels = torch.tensor([
                (sample['aux_data'].get('pronunciation_error') if sample['aux_data'] else 0)
                for sample in aux_per_sample
            ]).long()
            aux_targets['pronunciation_error'] = pron_labels

        return tuple(base_outputs + [aux_targets])

    def _uniform_frame_targets(self, text_tensor, mel_length):
        if mel_length <= 0:
            return torch.empty(0, dtype=torch.long)
        token_length = int(text_tensor.size(0))
        if token_length == 0:
            return torch.zeros(mel_length, dtype=torch.long)
        frame_targets = torch.zeros(mel_length, dtype=torch.long)
        ratio = mel_length / float(token_length)
        last_index = 0
        for token_idx in range(token_length):
            start = int(round(token_idx * ratio))
            end = int(round((token_idx + 1) * ratio))
            end = max(end, start + 1)
            end = min(end, mel_length)
            frame_targets[start:end] = text_tensor[token_idx]
            last_index = end
        if last_index < mel_length:
            frame_targets[last_index:] = text_tensor[-1]
        return frame_targets



def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     lengths=None,
                     bucket_sampler_config=None):

    auxiliary_objectives = dataset_config.get('auxiliary_objectives') if dataset_config else None
    if dataset_config and 'auxiliary_objectives' in dataset_config:
        dataset_config = {k: v for k, v in dataset_config.items() if k != 'auxiliary_objectives'}
    dataset_config = dict(dataset_config) if dataset_config else {}
    dataset = MelDataset(path_list, validation=validation, auxiliary_objectives=auxiliary_objectives, **dataset_config)
    collate_aux_objectives = collate_config.get('auxiliary_objectives') if collate_config else None
    include_aux = bool(collate_config.get('include_auxiliary_targets', False)) if collate_config else False
    if collate_config and 'auxiliary_objectives' in collate_config:
        collate_config = {k: v for k, v in collate_config.items() if k not in ('auxiliary_objectives', 'include_auxiliary_targets')}
    else:
        collate_config = dict(collate_config) if collate_config else {}
    collate_fn = Collater(auxiliary_objectives=collate_aux_objectives or auxiliary_objectives,
                          include_auxiliary_targets=include_aux,
                          **collate_config)

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
