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
pip install SoundFile torchaudio torch jiwer pyyaml click matplotlib librosa nltk pandas nvitop tensorboard accelerate
```
4. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training
```bash
python train.py --config_path ./Configs/config.yml
```
To utilise multiple GPUs, launch the trainer through [🤗 Accelerate](https://github.com/huggingface/accelerate):

```bash
accelerate launch --num_processes <num_gpus> train.py --config_path ./Configs/config.yml
```

The script automatically handles distributed setup, device placement and metric aggregation when run under `accelerate launch`.
The `batch_size` defined in `Configs/config.yml` represents the desired *global* batch size. When training across multiple GPUs
the launcher will automatically downscale the per-device batch size and keep the learning-rate scheduler in sync so that the
effective global batch matches the single-GPU configuration.
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
    enabled: true
    apply_prob: 0.5        # probability of applying SpecAugment on a batch
    policy: null           # set to LD or LF to use the large-delete / large-frequency presets
    freq_mask_param: 13
    time_mask_param: 50
    num_freq_masks: 2
    num_time_masks: 2
    time_warp:
      enabled: true
      window: 5
    adaptive_masking:
      enabled: true
      max_time_ratio: 0.15
      max_freq_ratio: 0.2
    random_frame_dropout:
      enabled: false
      drop_prob: 0.05
    vtlp:
      enabled: false
  waveform_augmentations:
    enabled: false
    noise:
      enabled: true
      gaussian: true
      snr_db: [10, 30]
    musan:
      enabled: false
      paths: []            # point this to the MUSAN noise folders to enable mixing
    reverberation:
      enabled: false
      rir_paths: []        # impulse responses to convolve with the waveform
    impulse_response:
      enabled: false
      paths: []
  mixup:
    enabled: false
    alpha: 0.4
    apply_prob: 0.2
  phoneme_dropout:
    enabled: false
    drop_prob: 0.1

loss_weights:
  ctc: 0.8                 # reduce to 0.5-0.7 for extremely small datasets
  s2s: 1.0
```

Each augmentation block can be toggled on or off independently through `Configs/config.yml`, allowing you to combine time warping, adaptive masking, frame dropping, VTLP, noise/reverberation mixing (including MUSAN and room impulse responses), and phoneme-level dropout as needed.

#### Keeping the CTC branch expressive

When the auxiliary ASR is trained on only a few hours of speech, the CTC head can collapse into predicting mostly blanks, which hurts diagonal alignments and increases the skip/merge gap. The default configuration now enables gentle blank-rate and coverage regularisers so deletions are discouraged without overwhelming the core losses:

```yaml
ctc_loss:
  blank_logit_bias: 0.0      # subtracts a constant from the blank logit before the softmax (kept neutral by default)
  logit_temperature: 1.0     # flattens the CTC posterior to discourage over-confidence
  regularization:
    blank_rate:
      enabled: true          # applies a mild hinge penalty when the blank posterior dominates
      weight: 0.25           # penalise batches whose blank rate exceeds ~0.62
      target: 0.62
      tolerance: 0.05        # slack before the penalty ramps up
    coverage:
      enabled: true          # encourages enough non-blank mass to cover the transcript length
      weight: 0.12
      margin: 4.0            # allows up to ~4 frames of under-coverage before the loss activates

alignment_regularization:
  attention_duration:
    enabled: true            # prevents the attention map from collapsing to 1–2 frame spikes
    weight: 0.15             # average penalty applied when tokens fall below the minimum duration
    min_frames: 2.0
    tolerance: 0.3

regularization:
  entropy:
    targets:
      ctc:
        weight: 0.02         # maximise entropy to keep non-blank symbols active
```

All of the auxiliary losses honour a `warmup_epochs` key (`5` for the CTC penalties and `8` for the duration term in the default config) so you can delay their activation until the alignment has roughly converged.

Decoding-time safeguards provide gentler blank suppression without distorting training. The beam-search configuration supports a temperature, blank penalty, insertion bonus, and lightweight length normalisation:

```yaml
decoding:
  beam_search:
    beam_width: 16
    length_penalty: 0.7
    logit_temperature: 1.1  # applied only during decoding
    blank_penalty: 0.05     # subtract from blank log-probabilities to reduce deletions
    insertion_bonus: 0.05   # encourages emitting non-blank symbols when hypotheses compete
```

Additionally, the diagonal attention prior now masks out padded timesteps, applies a light dropout (configurable through `model_params.attention_dropout`), and is scheduled via `diagonal_attention_prior_weight` so it ramps from a moderate early weight to a stronger late-epoch value. Coupled with the duration regulariser above, this keeps alignments monotonic while ensuring each phoneme retains a multi-frame footprint.

To avoid choosing checkpoints that only excel at PER while misaligning attention or dropping symbols, training now logs a joint selection score that blends PER, diagonal coherence, and the normalised CTC length gap. The configuration exposes the coefficients under `checkpoint_selection` and the trainer will keep a `best_joint.pth` symlink pointing at the most alignment-friendly checkpoint observed so far.

Increasing the warm-up ratio (`optimizer_params.pct_start`) or reducing the batch size can also help when the number of training utterances is limited.

### Languages
This repo is set up for English with the [phonemizer](https://github.com/bootphon/phonemizer) package and espeak-ng backend. You can train it with other languages. If you would like to train for datasets in different languages, you will need to change the vocabulary file ([word_index_dict.txt](https://github.com/martinambrus/AuxiliaryASR/blob/phonemizer/Data/word_index_dict.txt)) to contain the phonemes in your dataset. There is a utility script ([words_index_extractor.py](https://github.com/martinambrus/AuxiliaryASR/blob/phonemizer/words_index_extractor.py)) which will generate (and **_rewrite_**!) the file words_index_dict.txt when ran. Here's the syntax to use to generate this file:
```bash
python words_index_extractor.py --config_path ./Configs/config.yml
```

## References
- [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)

## Acknowledgement
The original author ([yl4579](https://github.com/yl4579/)) would like to thank [@tosaka-m](https://github.com/tosaka-m) for his great repository and valuable discussions.