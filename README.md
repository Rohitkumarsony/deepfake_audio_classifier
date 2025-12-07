# 🎵 Deepfake Audio Detection Model

A machine learning model to detect deepfake/synthetic audio using Wav2Vec2 embeddings and classical ML classifiers.

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/hjsgfd/deepfake_audio_classifier)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **92.86%** | 0.95 | 0.93 | 0.93 |
| SVM | 85.71% | 0.89 | 0.86 | 0.85 |
| Random Forest | 78.57% | 0.85 | 0.79 | 0.76 |

**Best Model: Logistic Regression with 92.86% accuracy**

## 🎯 Approach

### 1. Dataset
- **Source**: [Real vs Fake Human Voice Deepfake Audio Dataset](https://huggingface.co/datasets/ud-nlp/real-vs-fake-human-voice-deepfake-audio)
- **Size**: 70 audio samples
- **Classes**: 5 classes (0, 1, 2, 3, 4)
- **Distribution**: Perfectly balanced (14 samples per class)

### 2. Feature Extraction
We use **Wav2Vec2** (facebook/wav2vec2-base-960h) to extract deep audio embeddings:
- Pre-trained self-supervised model
- Extracts 768-dimensional feature vectors
- Captures semantic audio information
- Handles variable-length audio automatically

**Pipeline:**
```
Audio File → Wav2Vec2 → 768-dim Embedding → Classifier → Prediction
```

### 3. Model Training
Three classifiers were trained and compared:

#### Logistic Regression (Best)
- **Accuracy**: 92.86%
- Multi-class classification with OvR strategy
- Max iterations: 1000
- Features: StandardScaler normalized

#### SVM
- **Accuracy**: 85.71%
- RBF kernel
- Probability estimates enabled

#### Random Forest
- **Accuracy**: 78.57%
- 200 estimators
- Parallel processing enabled

### 4. Preprocessing
- **Audio Loading**: Support for both URLs and local files
- **Resampling**: All audio converted to 16kHz
- **Stereo to Mono**: Averaged across channels
- **Normalization**: StandardScaler on embeddings

## 🚀 Quick Start

### Installation
```bash
pip install transformers torch librosa soundfile scikit-learn huggingface-hub requests numpy
```

### Usage

#### Simple Prediction
```python
from predict_from_hf import AudioDeepfakeDetectorFromHF

# Initialize detector (downloads model automatically)
detector = AudioDeepfakeDetectorFromHF("hjsgfd/deepfake_audio_classifier")

# Predict from URL
result = detector.predict("https://your-audio-file.wav", is_url=True)
print(f"Prediction: {result['label']} ({result['confidence']:.1%})")
```

#### Batch Prediction
```python
from predict_from_hf import AudioDeepfakeDetectorFromHF

detector = AudioDeepfakeDetectorFromHF("hjsgfd/deepfake_audio_classifier")

# Multiple URLs
audio_urls = [
    "https://example.com/audio1.wav",
    "https://example.com/audio2.wav",
    "https://example.com/audio3.wav",
]

results = detector.predict_batch(audio_urls, are_urls=True)

# Print results
for result in results:
    if 'prediction' in result:
        print(f"{result['audio_source']}: {result['label']} ({result['confidence']:.1%})")
```

#### Local Files
```python
# Single file
result = detector.predict("path/to/audio.wav", is_url=False)

# Multiple files
local_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = detector.predict_batch(local_files, are_urls=False)
```

## 📁 Model Files

The model consists of three files hosted on Hugging Face:

1. **deepfake_audio_classifier.pkl** - Trained Logistic Regression classifier
2. **audio_scaler.pkl** - StandardScaler for feature normalization
3. **model_metadata.json** - Model configuration and metadata
```json
{
  "model_type": "LogisticRegression",
  "accuracy": 0.9286,
  "feature_extractor": "facebook/wav2vec2-base-960h",
  "embedding_dim": 768,
  "num_classes": 5,
  "class_labels": {
    "0": "class_0",
    "1": "class_1",
    "2": "class_2",
    "3": "class_3",
    "4": "class_4"
  }
}
```

## 📈 Detailed Results

### Training Configuration
- **Training Samples**: 56 (80%)
- **Testing Samples**: 14 (20%)
- **Feature Dimension**: 768
- **Stratified Split**: Maintains class distribution

### Logistic Regression Performance (Best Model)
```
              precision    recall  f1-score   support

     class_0       1.00      0.67      0.80         3
     class_1       1.00      1.00      1.00         2
     class_2       1.00      1.00      1.00         3
     class_3       0.75      1.00      0.86         3
     class_4       1.00      1.00      1.00         3

    accuracy                           0.93        14
   macro avg       0.95      0.93      0.93        14
weighted avg       0.95      0.93      0.93        14
```

### Key Metrics
- **Macro Average Precision**: 0.95
- **Macro Average Recall**: 0.93
- **Macro Average F1-Score**: 0.93
- **Overall Accuracy**: 92.86%

## 🔧 Technical Details

### Dependencies
```
transformers>=4.30.0
torch>=2.0.0
librosa>=0.10.0
soundfile>=0.12.0
scikit-learn>=1.3.0
huggingface-hub>=0.16.0
requests>=2.31.0
numpy>=1.24.0
```

### Model Architecture
```
Input: Audio File (any format supported by soundfile)
  ↓
Preprocessing (16kHz, Mono)
  ↓
Wav2Vec2 Feature Extractor
  ↓
768-dimensional Embedding
  ↓
StandardScaler Normalization
  ↓
Logistic Regression Classifier
  ↓
Output: Class Prediction + Confidence Scores
```

### Supported Audio Formats
- WAV
- MP3
- FLAC
- OGG
- M4A

## 📊 Training Process

1. **Data Loading**: Load dataset with disabled auto-decoding
2. **Feature Extraction**: Extract Wav2Vec2 embeddings (768-dim vectors)
3. **Train-Test Split**: 80-20 stratified split
4. **Normalization**: StandardScaler on training data
5. **Model Training**: Train 3 classifiers (LR, RF, SVM)
6. **Evaluation**: Compare performance on test set
7. **Selection**: Choose best model (Logistic Regression)
8. **Export**: Save model, scaler, and metadata

## 🎯 Use Cases

- Deepfake audio detection
- Voice authentication systems
- Media verification tools
- Forensic audio analysis
- Content moderation platforms

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Citation

If you use this model, please cite:
```bibtex
@misc{deepfake_audio_classifier_2024,
  author = {Your Name},
  title = {Deepfake Audio Detection Model},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/hjsgfd/deepfake_audio_classifier}}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: [ud-nlp/real-vs-fake-human-voice-deepfake-audio](https://huggingface.co/datasets/ud-nlp/real-vs-fake-human-voice-deepfake-audio)
- **Feature Extractor**: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Transformers Library**: Hugging Face

## 📧 Contact

For questions or feedback, please open an issue on the repository.

---
