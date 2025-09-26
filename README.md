# AuxiliaryASR
This repo contains the training code for Phoneme-level ASR for Voice Conversion (VC) and TTS (Text-Mel Alignment) used in [StyleTTS2](https://github.com/martinambrus/StyleTTS2). 

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/martinambrus/AuxiliaryASR.git
cd AuxiliaryASR
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio torch jiwer pyyaml click matplotlib librosa nltk pandas nvitop tensorboard
```
4. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training
```bash
python train.py --config_path ./Configs/config.yml
```
Please specify the training and validation data in `config.yml` file. The data list format needs to be `filename.wav|label-in-espeak-phonemes|speaker_number`, see [train_list.txt](https://github.com/martinambrus/AuxiliaryASR/blob/main/Data/train_list.txt) as an example (a custom sample of phonemized WAV files used for English training). Note that `speaker_number` can just be `0` for ASR, but it is useful to set a meaningful number for TTS training in StyleTTS2. 

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take - but not beyond 64, as anything beyond this value did not yield desired results in my testing. Please note that `batch_size = 64` will take around 10G GPU RAM.

### Length-aware batching

The default configuration now keeps batches balanced by grouping utterances with comparable durations. This prevents training from oscillating between very short and very long batches once SortaGrad switches to shuffled data. You can tweak or disable the sampler in `Configs/config.yml`:

```yaml
dataloader_params:
  train_bucket_sampler:
    enabled: true          # set to false to revert to the old uniform sampler
    bucket_size_multiplier: 20  # bucket = batch_size * multiplier
    shuffle_within_bucket: true # reshuffle items inside a bucket every epoch
    shuffle_batches: true       # randomize bucket order for each epoch
```

### Training on smaller datasets

Fine-tuning on very small corpora (a few hours of speech or less) tends to overfit quickly. The default `config.yml` now enables a light SpecAugment policy and a slightly smaller weight for the CTC branch (`loss_weights.ctc`). You can tweak those options to better fit your use case:

```yaml
dataset_params:
  spec_augment:
    apply_prob: 0.5    # probability of applying SpecAugment on a batch
    freq_mask_param: 13
    time_mask_param: 50
    num_freq_masks: 2
    num_time_masks: 2

loss_weights:
  ctc: 0.8             # reduce to 0.5-0.7 for extremely small datasets
  s2s: 1.0
```

Increasing the warm-up ratio (`optimizer_params.pct_start`) or reducing the batch size can also help when the number of training utterances is limited.

### Languages
The project now ships with a unified multilingual phoneme inventory that can emit either IPA or X-SAMPA symbols. The active standard and language overrides are controlled through `phoneme_settings` in `Configs/config.yml`. By default the configuration enables the IPA inventory defined in [`Configs/unified_phoneme_inventory.csv`](Configs/unified_phoneme_inventory.csv) and applies a set of reusable mappings from [`Configs/language_token_mappings.yml`](Configs/language_token_mappings.yml). During training the loader now inspects the metadata listed in `train_data`, `val_data`, and `ood_data`, extending the phoneme map with any previously unseen symbols before the first epoch. The resulting map is verified against [`Data/phoneme_map.txt`](Data/phoneme_map.txt) (which is generated automatically for new runs) so that accidental dataset changes or dictionary reorderings are caught early.

Every checkpoint now stores the vocabulary size it was trained with, letting the trainer abort early if a run is resumed with a mismatched phoneme inventory. If you intentionally change datasets, update or remove `Data/phoneme_map.txt` before starting a new run so the freshly derived map reflects your new data.

To adjust the behaviour you can toggle the individual options in the `phoneme_settings` block:

```yaml
phoneme_settings:
  enabled: true            # set to false to revert to the legacy per-language dictionaries
  mode: unified            # or 'legacy' to use a custom dictionary file
  standard: xsampa         # switch between 'ipa' and 'xsampa' without modifying the dataset
  apply_language_mappings: true
  active_mappings:         # choose which mapping groups should be active
    - identity_uppercase
    - identity_lowercase
    - arpabet_core
    - arpabet_extended
  allow_dynamic_extension: false  # enable to grow the vocabulary on the fly when encountering unseen tokens
```

You can extend the mapping file with additional language specific groups or provide dataset level overrides via `dataset_params.text_cleaner`. Legacy workflows that rely on a hand-crafted dictionary are still supported by setting `phoneme_settings.enabled` to `false` and pointing `phoneme_maps_path` (or `dataset_params.text_cleaner.dict_path`) at your vocabulary file.

## References
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## Acknowledgement
The original author ([yl4579](https://github.com/yl4579/)) would like to thank [@tosaka-m](https://github.com/tosaka-m) for his great repository and valuable discussions.