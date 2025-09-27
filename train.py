from meldataset import build_dataloader
from optimizers import build_optimizer
from utils import *
from models import build_model
from trainer import Trainer

import os
import os.path as osp
import math
import wave
import re
import sys
import yaml
import shutil
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import click
import importlib
import importlib.util

# enable better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


def _load_optional_tqdm():
    """Return tqdm.tqdm if available, otherwise ``None``."""
    spec = importlib.util.find_spec("tqdm")
    if spec is None:
        return None
    module = importlib.import_module("tqdm")
    return getattr(module, "tqdm", None)


_OPTIONAL_TQDM = _load_optional_tqdm()


def prepare_data_list(raw_data_list, root_path=""):
    """Parse metadata lines and compute WAV durations.

    Args:
        raw_data_list: Iterable with entries in the format
            ``path|phoneme sequence|speaker_id``.
        root_path: Base directory of the audio files.

    Returns:
        Tuple ``(prepared_list, durations)`` where ``prepared_list`` contains
        ``[path, text, speaker_id]`` entries and ``durations`` is a list with
        their corresponding durations (in seconds).
    """
    raw_data_sequence = list(raw_data_list)
    total_items = len(raw_data_sequence)
    prepared_list = []
    durations = []

    if total_items == 0:
        return prepared_list, durations

    progress_desc = "Computing audio durations"
    iterator = raw_data_sequence
    use_tqdm = _OPTIONAL_TQDM is not None

    if use_tqdm:
        iterator = _OPTIONAL_TQDM(raw_data_sequence,
                                  desc=progress_desc,
                                  total=total_items,
                                  unit="files")
    else:
        print(f"{progress_desc} for {total_items} files...")
        update_interval = max(1, total_items // 20)

    for index, line in enumerate(iterator):
        cleaned_line = line.rstrip('\n')
        if not cleaned_line.strip():
            continue

        parts = cleaned_line.split('|')
        if len(parts) < 2:
            print(f"Parse error for line: {cleaned_line}")
            continue

        path = parts[0].strip()
        if len(parts) == 2:
            text = parts[1]
            speaker_id = ""
        else:
            text = '|'.join(parts[1:-1])
            speaker_id = parts[-1].strip()

        wav_path = os.path.join(root_path, path)

        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate) if rate else 0.0
        except Exception as e:
            print(f"Error for wave path: {wav_path}, {e}")
            continue

        prepared_list.append([path, text, speaker_id])
        durations.append(duration)

        if not use_tqdm:
            should_update = ((index + 1) % update_interval == 0) or ((index + 1) == total_items)
            if should_update:
                print(f"Computed durations for {index + 1}/{total_items} files", flush=True)

    return prepared_list, durations


def sort_prepared_data_list(data_list, durations):
    if len(data_list) != len(durations):
        raise ValueError("data_list and durations must have the same length")

    paired = sorted(zip(durations, data_list), key=lambda x: x[0])
    sorted_list = [item for _, item in paired]
    sorted_durations = [duration for duration, _ in paired]
    return sorted_list, sorted_durations


def sort_data_list_by_duration(raw_data_list=None, root_path="", precomputed=None):
    """
    Sort metadata entries by ascending audio duration.

    Args:
        raw_data_list: Iterable with raw metadata lines. Optional when
            ``precomputed`` is provided.
        root_path: Base directory of the audio files (used when parsing the
            raw metadata).
        precomputed: Optional tuple ``(prepared_list, durations)`` as returned
            by :func:`prepare_data_list`.

    Returns:
        A tuple ``(sorted_list, sorted_durations)``.
    """
    if precomputed is not None:
        data_list, durations = precomputed
    else:
        if raw_data_list is None:
            raise ValueError("Either raw_data_list or precomputed must be provided")
        data_list, durations = prepare_data_list(raw_data_list, root_path=root_path)

    return sort_prepared_data_list(data_list, durations)

def cfg_get_nested(cfg: dict, path, default=None, sep="."):
    """
    Get a nested value from a dict using a list of keys or a dot-separated string.

    Examples:
        cfg_get_nested(config, ["model_params", "input_dim"], 80)
        cfg_get_nested(config, "model_params.input_dim", 80)
        cfg_get_nested(config, "top_key", 80)
    """
    if isinstance(path, str):
        keys = path.split(sep)
    else:
        keys = path

    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

class EarlyStoppingWithNoLearningRate:
    def __init__(self, patience=5):
        self.patience = patience  # Number of epochs to wait for improvement
        self.counter = 0
        self.stop_training = False

    def __call__(self, value):
        if round( value, 5 ) <= 0.0000:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True

        return self.stop_training

@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):
    print(f"Loading config data from: {config_path}")
    config = yaml.safe_load(open(config_path))

    log_dir = cfg_get_nested( config, 'log_dir', 'Checkpoint' )
    print(f"Using logs and models folder: {log_dir}")

    if not osp.exists(log_dir): os.mkdir(log_dir)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = cfg_get_nested( config, 'batch_size', 10)
    device = cfg_get_nested( config, 'device', 'cpu')
    epochs = cfg_get_nested( config, 'epochs', 200)
    save_freq = cfg_get_nested( config, 'save_freq', 10)
    train_path = cfg_get_nested( config, 'train_data', None)
    val_path = cfg_get_nested( config, 'val_data', None)
    enable_early_stopping = cfg_get_nested( config, 'enable_early_stopping', True)
    dataset_params = {
        'dict_path': cfg_get_nested( config, 'phoneme_maps_path', 'Data/word_index_dict.txt'),
        'sr': cfg_get_nested( config, 'preprocess_params.sr', 24000),
        'spect_params': cfg_get_nested( config, 'preprocess_params.spect_params', {
            'n_fft': 1024,
            'win_length': 1024,
            'hop_length': 300
        }),
        'mel_params': cfg_get_nested( config, 'preprocess_params.mel_params', { 'n_mels': 80 })
    }

    dataset_additional_params = cfg_get_nested(config, 'dataset_params', {})
    if isinstance(dataset_additional_params, dict):
        for override_key in ('dict_path', 'sr', 'spect_params', 'mel_params'):
            if override_key in dataset_additional_params:
                dataset_params[override_key] = dataset_additional_params[override_key]

        if 'spec_augment' in dataset_additional_params:
            dataset_params['spec_augment_params'] = dataset_additional_params['spec_augment']

        for override_key, override_value in dataset_additional_params.items():
            if override_key in ('dict_path', 'sr', 'spect_params', 'mel_params', 'spec_augment'):
                continue
            dataset_params[override_key] = override_value

    raw_train_list, raw_val_list = get_data_path_list(train_path, val_path)

    train_entries, train_durations = prepare_data_list(raw_train_list, root_path="")
    val_entries, val_durations = prepare_data_list(raw_val_list, root_path="")

    # sort data list by duration for consistent bucketing and batching
    # to reduce padding, improving both training stability and alignment accuracy
    train_list_sorted, _ = sort_data_list_by_duration(precomputed=(train_entries, train_durations))
    val_list_sorted, _ = sort_data_list_by_duration(precomputed=(val_entries, val_durations))

    dataloader_params = cfg_get_nested(config, 'dataloader_params', {})
    train_num_workers = int(dataloader_params.get('train_num_workers', 8))
    val_num_workers = int(dataloader_params.get('val_num_workers', 2))
    train_bucket_sampler_config = dataloader_params.get('train_bucket_sampler', {})

    collate_config = {'return_speaker_ids': True}

    sorted_train_dataloader = build_dataloader(train_list_sorted,
                                        batch_size=batch_size,
                                        num_workers=train_num_workers,
                                        dataset_config=dataset_params,
                                        device=device,
                                        collate_config=collate_config)
    shuffled_train_dataloader = build_dataloader(train_entries,
                                            batch_size=batch_size,
                                            num_workers=train_num_workers,
                                            dataset_config=dataset_params,
                                            device=device,
                                            lengths=train_durations,
                                            bucket_sampler_config=train_bucket_sampler_config,
                                            collate_config=collate_config)

    sorted_val_dataloader = build_dataloader(val_list_sorted,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=val_num_workers,
                                      device=device,
                                      dataset_config=dataset_params,
                                      collate_config=collate_config)
    shuffled_val_dataloader = build_dataloader(val_entries,
                                          batch_size=batch_size,
                                          validation=True,
                                          num_workers=val_num_workers,
                                          device=device,
                                          dataset_config=dataset_params,
                                          collate_config=collate_config)

    word_indexes = set(
        line.strip() for line in open(cfg_get_nested( config, 'phoneme_maps_path', 'Data/word_index_dict.txt'))
        if line.strip()
    )

    model_params = cfg_get_nested( config, 'model_params', {
        'input_dim': 80,
        'hidden_dim': 256,
        'n_token': len( word_indexes ),
        'token_embedding_dim': 512,
        'n_layers': 5,
        'location_kernel_size': 31
    })

    if not 'n_token' in model_params:
        model_params['n_token'] = len( word_indexes )

    multi_task_config = cfg_get_nested(config, 'multi_task', {}) or {}

    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    if speaker_cfg.get('enabled', False) and int(speaker_cfg.get('num_speakers', 0)) <= 0:
        speaker_ids = set()
        for entry in train_entries + val_entries:
            if len(entry) >= 3:
                speaker_id = str(entry[2]).strip()
                if speaker_id:
                    speaker_ids.add(speaker_id)
        inferred = max(1, len(speaker_ids))
        print(f"Inferred {inferred} unique speaker id(s) from metadata")
        speaker_cfg = dict(speaker_cfg)
        speaker_cfg['num_speakers'] = inferred
        multi_task_config['speaker'] = speaker_cfg

    stabilization_config = cfg_get_nested(config, 'stabilization', {}) or {}

    model_params = dict(model_params)
    model_params['multi_task_config'] = multi_task_config
    model_params['stabilization_config'] = stabilization_config

    print("Using model parameters:", model_params)

    model = build_model(model_params=model_params)

    scheduler_params = {
            'max_lr': float(cfg_get_nested( config, 'optimizer_params.lr', 5e-4)),
            'pct_start': float(cfg_get_nested( config, 'optimizer_params.pct_start', 0.1)),
            'epochs': epochs,
            'steps_per_epoch': len(sorted_train_dataloader),
        }

    entropy_params = cfg_get_nested( config, 'entropy_params', { "label_smoothing": 0.1 })

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})

    blank_index = sorted_train_dataloader.dataset.text_cleaner.word_index_dictionary[" "] # get blank index
    criterion = build_criterion(critic_params={
                'ctc': {'blank': blank_index, 'reduction': 'none', 'zero_infinity': True},
        }, entropy_params=entropy_params, multi_task_config=multi_task_config)

    if enable_early_stopping:
        early_stopping = EarlyStoppingWithNoLearningRate(patience=max([ 3, int( math.floor( int( cfg_get_nested( config, 'save_freq', 10 ) ) / 2 ) ) ]) )
    else:
        early_stopping = None

    loss_weight_config = cfg_get_nested(config, 'loss_weights', {}) or {}
    regularization_config = cfg_get_nested(config, 'regularization', {}) or {}
    entropy_regularization_config = cfg_get_nested(regularization_config, 'entropy', {}) or {}
    use_ctc = bool(multi_task_config.get('use_ctc', True))
    use_s2s = bool(multi_task_config.get('use_seq2seq', True))
    frame_cfg = multi_task_config.get('frame_phoneme', {}) or {}
    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    pron_cfg = multi_task_config.get('pronunciation_error', {}) or {}

    ctc_weight = float(loss_weight_config.get('ctc', 1.0 if use_ctc else 0.0)) if use_ctc else 0.0
    s2s_weight = float(loss_weight_config.get('s2s', 1.0 if use_s2s else 0.0)) if use_s2s else 0.0
    frame_weight = float(loss_weight_config.get('frame_phoneme', 0.0)) if frame_cfg.get('enabled', False) else 0.0
    speaker_weight = float(loss_weight_config.get('speaker', 0.0)) if speaker_cfg.get('enabled', False) else 0.0
    pron_weight = float(loss_weight_config.get('pronunciation_error', 0.0)) if pron_cfg.get('enabled', False) else 0.0
    mixspeech_config = stabilization_config.get('mix_speech', {}) or {}
    intermediate_ctc_config = stabilization_config.get('intermediate_ctc', {}) or {}
    if isinstance(intermediate_ctc_config, dict) and 'loss_weight' not in intermediate_ctc_config:
        intermediate_ctc_config = dict(intermediate_ctc_config)
        intermediate_ctc_config['loss_weight'] = float(loss_weight_config.get('intermediate_ctc', 0.0))
    self_conditioned_ctc_config = stabilization_config.get('self_conditioned_ctc', {}) or {}
    if isinstance(self_conditioned_ctc_config, dict) and 'loss_weight' not in self_conditioned_ctc_config:
        self_conditioned_ctc_config = dict(self_conditioned_ctc_config)
        self_conditioned_ctc_config['loss_weight'] = float(loss_weight_config.get('self_conditioned_ctc', 0.0))

    steps_per_epoch = len(sorted_train_dataloader) if sorted_train_dataloader is not None else len(shuffled_train_dataloader)

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    device=device,
                    sorted_train_dataloader=sorted_train_dataloader,
                    shuffled_train_dataloader=shuffled_train_dataloader,
                    sorted_val_dataloader=sorted_val_dataloader,
                    shuffled_val_dataloader=shuffled_val_dataloader,
                    logger=logger,
                    switch_sortagrad_dataset_epoch=cfg_get_nested( config, 'sortagrad_switch_to_shuffled_dataset_epoch', 10),
                    use_diagonal_attention_prior=(cfg_get_nested( config, 'use_diagonal_attention_prior', True) and use_s2s),
                    diagonal_attention_prior_weight=cfg_get_nested( config, 'diagonal_attention_prior_weight', 0.1),
                    ctc_weight=ctc_weight,
                    s2s_weight=s2s_weight,
                    frame_weight=frame_weight,
                    speaker_weight=speaker_weight,
                    pron_error_weight=pron_weight,
                    enable_frame_classifier=frame_cfg.get('enabled', False),
                    enable_speaker=speaker_cfg.get('enabled', False),
                    enable_pronunciation_error=pron_cfg.get('enabled', False),
                    mixspeech_config=mixspeech_config,
                    intermediate_ctc_config=intermediate_ctc_config,
                    self_conditioned_ctc_config=self_conditioned_ctc_config,
                    entropy_regularization_config=entropy_regularization_config,
                    steps_per_epoch=steps_per_epoch
                    )

    pretrained_model = cfg_get_nested( config, 'pretrained_model', '' )
    load_only_params = cfg_get_nested( config, 'load_only_params', False )
    if isinstance(pretrained_model, bool) and pretrained_model == True:
        try:
            files = os.listdir(log_dir)
            ckpts = []
            for f in os.listdir(log_dir):
                if f.startswith("epoch_") and f.endswith(".pth"): ckpts.append(f)

            if len(ckpts):
                iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
                iters = sorted(iters)[-1]

                checkpoint_file = log_dir + f"/epoch_{iters:05}.pth"
                print(f"Starting to train from checkpoint {checkpoint_file}")
                start_epoch = trainer.load_checkpoint(checkpoint_file, load_only_params=load_only_params)
            else:
                print(f"No previous checkpoints found, starting training from epoch 1.")
                start_epoch = 1
        except Exception as e:
            print(f"Failed to load latest checkpoint, starting training from epoch 1 - {e}")
            start_epoch = 1
    elif isinstance(pretrained_model, str) and pretrained_model != "":
        start_epoch = trainer.load_checkpoint(pretrained_model, load_only_params=load_only_params)
        start_epoch += 1
        print(f"Checkpoint {pretrained_model} loaded, starting training from epoch {start_epoch}.")
    elif ( isinstance(pretrained_model, str) and pretrained_model == "" ) or ( isinstance(pretrained_model, bool) and pretrained_model == False ):
        print(f"Starting training from epoch 1.")
        start_epoch = 1
    else:
        print(f"Unrecognized value for load_checkpoint config option, starting training from epoch 1 - {pretrained_model}")
        start_epoch = 1

    for epoch in range(start_epoch, epochs+1):
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()

        # Get learning rate from training results
        learning_rate = train_results.get('train/learning_rate', None)  # Ensure 'eval/cer' exists in your eval_results

        if learning_rate is None:
            raise Exception("learning_rate not found in training results! Please check the metric calculation.")
            continue  # Skip if CER is missing

        results = train_results.copy()
        results.update(eval_results)
        logger.info('--- epoch %d ---' % epoch)
        for key, value in results.items():
            if isinstance(value, float):
                logger.info('%-15s: %.5f' % (key, value))
                writer.add_scalar(key, value, epoch)
            else:
                for v in value:
                    writer.add_figure('eval_attn', plot_image(v), epoch)

        if (epoch % save_freq) == 0 or ( early_stopping != None and early_stopping(learning_rate) ):
            trainer.save_checkpoint(osp.join(log_dir, 'epoch_%05d.pth' % epoch))

        # Check if early stopping condition is met
        if early_stopping != None and early_stopping(learning_rate):
            logger.info(f"Early stopping triggered at epoch {epoch}, learning_rate: {learning_rate:.5f}")
            break

    return 0

if __name__=="__main__":
    main()