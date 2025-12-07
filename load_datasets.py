from datasets import load_dataset
# Load dataset with audio decoding explicitly disabled
def load_data():
    ds = load_dataset(
        "ud-nlp/real-vs-fake-human-voice-deepfake-audio",
        split="train"
    )
    return ds