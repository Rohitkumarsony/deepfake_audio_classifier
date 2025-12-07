import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# Assume X and y are already loaded from your previous code
# X shape: (70, 768)
# y shape: (70,)

def train_and_save_model(X, y):
    """
    Train classifier on extracted embeddings and save the model
    
    Args:
        X: Embeddings array (n_samples, n_features)
        y: Labels array (n_samples,)
    """
    
    print("="*60)
    print("DEEPFAKE AUDIO DETECTION - MODEL TRAINING")
    print("="*60)
    
    print(f"\nDataset Info:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Train-test split
    print("\n[1/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Testing samples: {len(X_test)}")
    
    # Standardize features
    print("\n[2/4] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized")
    
    # Train multiple models
    print("\n[3/4] Training models...")
    
    models = {}
    
    # Logistic Regression
    print("\n" + "-"*60)
    print("Training Logistic Regression...")
    print("-"*60)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)
    models['LogisticRegression'] = {
        'model': lr_model,
        'accuracy': lr_accuracy,
        'predictions': y_pred_lr
    }
    
    print(f"Accuracy: {lr_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    # Random Forest
    print("\n" + "-"*60)
    print("Training Random Forest...")
    print("-"*60)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    models['RandomForest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': y_pred_rf
    }
    
    print(f"Accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # SVM
    print("\n" + "-"*60)
    print("Training SVM...")
    print("-"*60)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    models['SVM'] = {
        'model': svm_model,
        'accuracy': svm_accuracy,
        'predictions': y_pred_svm
    }
    
    print(f"Accuracy: {svm_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_svm))
    
    # Select best model
    print("\n[4/4] Selecting best model...")
    best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
    best_model = models[best_model_name]['model']
    best_accuracy = models[best_model_name]['accuracy']
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"ACCURACY: {best_accuracy:.4f}")
    print("="*60)
    
    # Model comparison
    print("\nModel Comparison:")
    for name, info in models.items():
        print(f"  {name:20s}: {info['accuracy']:.4f}")
    
    # Save model
    print("\nSaving model components...")
    
    joblib.dump(best_model, 'deepfake_audio_classifier.pkl')
    print("✓ Model saved: deepfake_audio_classifier.pkl")
    
    joblib.dump(scaler, 'audio_scaler.pkl')
    print("✓ Scaler saved: audio_scaler.pkl")
    
    # Create label mapping
    unique_labels = sorted(np.unique(y))
    label_mapping = {int(label): f'class_{label}' for label in unique_labels}
    
    metadata = {
        'model_type': best_model_name,
        'accuracy': float(best_accuracy),
        'feature_extractor': 'facebook/wav2vec2-base-960h',
        'embedding_dim': int(X.shape[1]),
        'num_classes': int(len(unique_labels)),
        'class_labels': label_mapping,
        'all_models_accuracy': {name: float(info['accuracy']) for name, info in models.items()}
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✓ Metadata saved: model_metadata.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE! ✨")
    print("="*60)
    
    return best_model, scaler, metadata

