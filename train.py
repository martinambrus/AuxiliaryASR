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


def sort_data_list_by_duration(raw_data_list, root_path="/"):
    """
    raw_data_list: list of strings like "path/to.wav|some text|speaker_id"
    root_path:     base directory for your wav files

    returns: list of tuples (path, text, speaker_id),
             sorted by ascending duration
    """
    durations = []
    for line in raw_data_list:
        try:
            path, text, speaker_id = line.strip().split('|')
            wav_path = os.path.join(root_path, path)
        except Exception as e:
            print(f"Parse error for line: {line}, {e}")
            continue

        # read header only (fast)
        try:
            with wave.open(wav_path, 'rb') as wf:
                frames = wf.getnframes()
                rate   = wf.getframerate()
                dur    = frames / float(rate)

            durations.append((dur, path, text, speaker_id))
        except Exception as e:
            print(f"Error for wave path: {wav_path}, {e}")

    # sort by duration
    durations.sort(key=lambda x: x[0])

    # drop the duration, return only the data triples
    #sorted_list = [(path, text, speaker_id) for _, path, text, speaker_id in durations]
    sorted_list = [[path, text, speaker_id] for _, path, text, speaker_id in durations]
    return sorted_list

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

    train_list, val_list = get_data_path_list(train_path, val_path)

    # sort data list by duration for consistent bucketing and batching
    # to reduce padding, improving both training stability and alignment accuracy
    train_list_sorted = sort_data_list_by_duration(train_list, root_path="")
    val_list_sorted = sort_data_list_by_duration(val_list, root_path="")

    # split original train and val lists CSV lines by |
    train_list = [l[:-1].split('|') for l in train_list]
    val_list = [l[:-1].split('|') for l in val_list]

    sorted_train_dataloader = build_dataloader(train_list_sorted,
                                        batch_size=batch_size,
                                        num_workers=8,
                                        dataset_config=dataset_params,
                                        device=device)
    shuffled_train_dataloader = build_dataloader(train_list,
                                            batch_size=batch_size,
                                            num_workers=8,
                                            dataset_config=dataset_params,
                                            device=device)

    sorted_val_dataloader = build_dataloader(val_list_sorted,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config=dataset_params)
    shuffled_val_dataloader = build_dataloader(val_list,
                                          batch_size=batch_size,
                                          validation=True,
                                          num_workers=2,
                                          device=device,
                                          dataset_config=dataset_params)

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
    model = build_model(model_params=model_params)

    scheduler_params = {
            'max_lr': float(cfg_get_nested( config, 'optimizer_params.lr', 5e-4)),
            'pct_start': float(cfg_get_nested( config, 'optimizer_params.pct_start', 0.0)),
            'epochs': epochs,
            'steps_per_epoch': len(sorted_train_dataloader),
        }

    entropy_params = cfg_get_nested( config, 'entropy_params', { "label_smoothing": 0.1 })

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})

    blank_index = sorted_train_dataloader.dataset.text_cleaner.word_index_dictionary[" "] # get blank index
    criterion = build_criterion(critic_params={
                'ctc': {'blank': blank_index},
        }, entropy_params=entropy_params)

    if enable_early_stopping:
        early_stopping = EarlyStoppingWithNoLearningRate(patience=max([ 3, int( math.floor( int( cfg_get_nested( config, 'save_freq', 10 ) ) / 2 ) ) ]) )
    else:
        early_stopping = None

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    sorted_train_dataloader=sorted_train_dataloader,
                    shuffled_train_dataloader=shuffled_train_dataloader,
                    sorted_val_dataloader=sorted_val_dataloader,
                    shuffled_val_dataloader=shuffled_val_dataloader,
                    logger=logger,
                    switch_sortagrad_dataset_epoch=cfg_get_nested( config, 'sortagrad_switch_to_shuffled_dataset_epoch', 10),
                    use_diagonal_attention_prior=cfg_get_nested( config, 'use_diagonal_attention_prior', True),
                    diagonal_attention_prior_weight=cfg_get_nested( config, 'diagonal_attention_prior_weight', 0.1),
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