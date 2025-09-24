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

### Auxiliary variance adaptor for alignment

For TTS alignment use cases you can enable an auxiliary FastSpeech-style variance adaptor that predicts token durations from the decoder embeddings. This stabilizes attention learning by providing an extra supervision signal. Configure it from `Configs/config.yml`:

```yaml
model_params:
  variance_adaptor:
    enabled: true           # instantiate the duration predictor
    filter_size: 256        # convolution width inside the predictor
    kernel_size: 3
    dropout: 0.5
    num_layers: 2
    predict_log_duration: true

loss_weights:
  duration: 0.2             # weight of the auxiliary duration loss

auxiliary_alignment:
  duration_loss:
    enabled: true           # toggle the auxiliary loss without touching weights
    target_strategy: uniform  # or uniform_rounded / attention_peaks
    log_target: true         # compare in the log-domain (recommended)
    mask_padding: true       # ignore padded tokens when computing the loss
```

The uniform target strategy distributes the acoustic frames evenly across phonemes, while `attention_peaks` derives coarse durations from the decoder attention map. You can disable the adaptor entirely by setting `variance_adaptor.enabled` to `false` or keep it enabled while zeroing out the loss weight for diagnostics.

### Languages
This repo is set up for English with the [phonemizer](https://github.com/bootphon/phonemizer) package and espeak-ng backend. You can train it with other languages. If you would like to train for datasets in different languages, you will need to change the vocabulary file ([word_index_dict.txt](https://github.com/martinambrus/AuxiliaryASR/blob/main/word_index_dict.txt)) to contain the phonemes in your dataset. There is a utility script ([words_index_extractor.py](https://github.com/martinambrus/AuxiliaryASR/blob/main/words_index_extractor.py)) which will generate (and **_rewrite_**!) the file words_index_dict.txt when ran. Here's the syntax to use to generate this file:
```bash
python words_index_extractor.py --config_path ./Configs/config.yml
```

## References
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## Acknowledgement
The original author ([yl4579](https://github.com/yl4579/)) would like to thank [@tosaka-m](https://github.com/tosaka-m) for his great repository and valuable discussions.