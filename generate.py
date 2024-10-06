import os
import re
import string
import argparse
import warnings
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import soundfile as sf
from speechbrain.inference.enhancement import WaveformEnhancement
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from utils import REPLACEMENTS, SAMPLE_TEXT

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_filename", help="name of the output file", type=str, default="output_audio")
    parser.add_argument("--speaker", help="'f' for female or 'm' for male", type=str, default="f")
    parser.add_argument("--device", help="cpu or gpu", type=str, default='cpu')
    parser.add_argument("--text", help="input text", type=str, default=SAMPLE_TEXT)
    args = parser.parse_args()

    if args.output_filename:
        OUTPUT_NAME = args.output_filename
        if not OUTPUT_NAME.isalnum():
            raise ValueError("Output filename must contain only alphanumeric characters.")

    if args.speaker:
        if args.speaker == "f":
            speaker_id = 27
        elif args.speaker == "m":
            speaker_id = 29
        else:
            raise ValueError("Speaker not defined. Use 'f' for female or 'm' for male.")
        
    if args.text:
        INPUT_TEXT = args.text

    if args.device:
        if args.device == "cpu":
            device = torch.device('cpu')
        elif args.device == "gpu":
            if torch.cuda.is_available():
                device = torch.device('cuda') 
            else:
                raise ValueError("GPU CUDA device not available.")
        else:
            raise ValueError("Device not defined. Use 'cpu' or 'gpu'.")
        
    model = SpeechT5ForTextToSpeech.from_pretrained("nikolab/speecht5_tts_hr").to(device)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    def cleanup_text(input_text):
        for src, dst in REPLACEMENTS:
            input_text = input_text.replace(src, dst)
        pattern = r'[{}]'.format(re.escape(string.punctuation))
        result = re.sub(pattern, '', input_text)
        return result

    def chunking_by_sentences(input_text, max_words=10):
        sentences = re.split(r'(?<=[.!?])', input_text)
        sentences = list(filter(None, sentences))
        chunks = []
        for sentence_text in sentences:
            words = sentence_text.split()
            sentence = ""
            for word in words:
                if len(sentence.split()) < max_words:
                    sentence += word + " "
                else:
                    chunks.append(sentence.strip())
                    sentence = word + " "
            if sentence:
                chunks.append(sentence.strip())
        return chunks

    chunks = chunking_by_sentences(input_text=INPUT_TEXT, max_words=12)
    speaker_embeddings = torch.load("./speaker_embeddings/speaker_embeddings_test.pt")[speaker_id].unsqueeze(0)
    resulting_audio = np.array([])
    print("Generating audio...")
    for chunk in tqdm(chunks):
        inputs = processor(text=cleanup_text(chunk), return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"].to(device), 
                                    speaker_embeddings.to(device), 
                                    vocoder=vocoder)
        resulting_audio = np.concatenate([resulting_audio, speech.cpu().numpy()])

    sf.write(f"audio_temp.wav", resulting_audio, samplerate=16000)
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
        run_opts={"device":device}
    )
    enhanced = enhance_model.enhance_file("./audio_temp.wav")
    torchaudio.save(f'{OUTPUT_NAME}.wav', enhanced.unsqueeze(0).cpu(), 16000)
    os.remove("./audio_temp.wav")
    print("Audio saved as audio_result.wav")
