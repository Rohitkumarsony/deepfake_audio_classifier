from datasets import load_dataset, Audio
import soundfile as sf
import io
import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
from load_datasets import load_data
from model_train import train_and_save_model
# Disable audio decoding by casting with decode=False
ds=load_data()
ds = ds.cast_column("audio", Audio(decode=False))

# Load Wav2Vec2
extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")

def decode_audio(example):
    audio_data = example["audio"]
    
    # Check if 'bytes' key exists, otherwise use 'path'
    if "bytes" in audio_data and audio_data["bytes"] is not None:
        audio_bytes = audio_data["bytes"]
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    elif "path" in audio_data and audio_data["path"] is not None:
        # If bytes not available, read from path
        audio_array, sr = sf.read(audio_data["path"])
    else:
        raise ValueError("No valid audio data found")
    
    audio_array = audio_array.astype("float32")
    return audio_array, sr

def get_embedding(audio_array, sr):
    # Handle stereo to mono conversion
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        sr = 16000

    inputs = extractor(audio_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.squeeze().numpy()

X, y = [], []

print("Extracting embeddings...")

# First, let's check the structure of the first example
print("Checking first example structure...")
first_example = ds[0]
print("Audio keys:", first_example["audio"].keys() if isinstance(first_example["audio"], dict) else type(first_example["audio"]))

for example in tqdm(ds):
    try:
        audio_array, sr = decode_audio(example)
        label = example["label"]
        emb = get_embedding(audio_array, sr)
        X.append(emb)
        y.append(label)
    except Exception as e:
        print(f"\nError processing example: {e}")
        continue

X = np.array(X)
y = np.array(y)

train_and_save_model(X, y)
print("\nEmbeddings Shape:", X.shape)
print("Labels Shape:", y.shape)
print("Unique labels:", np.unique(y))

