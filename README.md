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