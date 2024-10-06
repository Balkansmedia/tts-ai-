# tts-ai

TTS-AI is open-source text-to-speech AI project tailored for ex-yu (BCSM) languages.

# Installation

To install conda environment, run the following command:
```bash
conda create -n "env_name" python=3.11.9 ipython
```

and then activate the environment:
```bash
conda activate "env_name"
```

After the environment is activated, install the required packages:
```bash
pip install -r requirements.txt
```

# Usage

Speech generation can be triggered by running ```generate.py``` script with appropriate arguments. The script will generate speech from the input text and save it to the output file.

Example usage:
```bash
python generate.py --output_filename "output_audio" --speaker="f" --device="cpu" --text="Ovo je primjer generisanog zvučnog snimka pomoću sistema vještačke inteligencije."
```

Arguments:
- ```output_filename``` - name of the output audio file (alphanumeric and without extension)
- ```speaker``` - speaker (```m``` for male, ```f``` for female)
- ```device``` - device on which the model will run (```cpu``` or ```gpu```)
- ```text``` - input text for speech generation

# Model

SpeechT5 is a fine-tuned model for speech synthesis (text-to-speech) on the LibriTTS dataset. It was created as an upgraded version of the successful T5 model (Text-To-Text Transfer Transformer), which was trained only for natural language processing. The model was originally presented by the Microsoft research group, in the scientific paper "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing" (https://arxiv.org/abs/2110.07205).

The SpeechT5 model was also chosen due to the extensive evaluation, carried out in the mentioned scientific work, which showed very good results in a wide range of speech processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement and speaker identification.

SpeechT5 contains three types of speech models in one architecture. The model can be used to convert:
- speech to text - for automatic speech recognition or speaker identification,
- text to speech - for sound synthesis,
- speech to speech - to convert between different voices or improve speech.

The SpeechT5 model consists of a common network of encoders and decoders, with an additional six neural networks that are specific to the particular modality of the data being processed (speech/text). The unique thing about the SpeechT5 model is that the model is first pre-trained on different speech-to-text and text-to-speech data modalities, so that it learns in the unified representation space of both text and speech. In this way, the model learns from text and speech at the same time. This allows us to fine-tune the pre-trained model for different tasks, such as text-to-speech, in ex-yu languages (Montenegrin, Serbian, Bosnian, Croatian).

# Dataset

LibriTTS (https://www.openslr.org/60/) is a multi-speaker English corpus of approximately 585 hours of English speech, prepared by Heiga Zen with the help of members of the Google Speech and Google Brain teams. This corpus is designed for TTS (text-to-speech) research. It is derived from the original LibriSpeech corpus (https://www.openslr.org/12/) - mp3 audio files from LibriVox and text files from Project Gutenberg.

The VoxPopuli dataset, published in the scientific paper link, contains:
- 400 thousand hours of untagged voice data for 23 languages,
- 1.8 thousand hours of transcribed speech data for 16 languages,
- 17.3 thousand hours of "speech-to-speech" data,
- 29 hours of transcribed speech data of non-native English speakers, intended for research into accented speech.

# Technical implementation

Experimental trainings of the SpeechT5 model were carried out with the aim of adapting the basic model for the use of text-to-speech conversion.

As the original SpeechT5 model was trained on tasks exclusively in English (LibriTTS dataset), it was necessary to implement the training of the new model, on the available data originating from a group of regional languages ​​(Montenegrin, Serbian, Bosnian, Croatian). One of the popular open datasets for this use is the VoxPopuli set, which contains sound recordings of the European Parliament from 2009 to 2020. Given that data in all regional languages ​​is not available to the required extent, data in the Croatian language, which is the most represented, was taken from the VoxPopuli dataset. In the next stages of the project, data will be collected in Montenegrin, Serbian and Bosnian languages, in order to improve the quality of training and the accuracy of the model.

Thus, the final dataset consists of 43 transcribed hours of speech, 83 different speakers and 337 thousand transcribed tokens (1 token = 3/4 words).

In the first phase of technical implementation, the dataset went through several stages of processing in order to adapt and standardize it for training the SpeechT5 model. Data processing methods belong to the standard methods of linguistic data manipulation in the field of natural language processing (vocabulary formation, tokenization, removal or conversion of unsupported characters/letters, text/speech cleaning, text normalization).

In the next phase, the statistics of speakers in the VoxPopuli dataset were analyzed, based on which speakers with satisfactory text/speech quality and a sufficient number of samples for model training were selected. In this phase, the balancing of the dataset was carried out so that both male and female speakers, with high-quality text/speech samples, were equally represented in the training.

After the preparation of the data, the adjustment and optimization of the hyperparameters of the SpeechT5 model, which are necessary so that the training of the model can be performed quickly and efficiently, with satisfactory accuracy, was started. Several experimental training sessions were performed to obtain the optimal hyperparameters, which were then used in the evaluation phase of the model.

The evaluation of the model, on the dataset intended for testing, showed promising results. The model started to learn on the prepared dataset, but it also showed certain limitations. The main limitation is related to the length of the input text sequence. The model showed the inability to generate speech for long sequences of input text (over 20 words). The limitation was overcome by dividing the input sequence into smaller units and in that form passed to the model for processing. The main reason for the appearance of this limitation lies primarily in the lack of a large amount of data on which it is necessary to fine-tune the model in order to obtain the best possible results.


