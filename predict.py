# After your existing code that creates X and y...

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

print("="*60)
print("TRAINING CLASSIFIER")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {}

# Logistic Regression
print("\n[1/3] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
models['LogisticRegression'] = {'model': lr, 'accuracy': lr_acc}
print(f"Accuracy: {lr_acc:.4f}")

# Random Forest
print("\n[2/3] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
models['RandomForest'] = {'model': rf, 'accuracy': rf_acc}
print(f"Accuracy: {rf_acc:.4f}")

# SVM
print("\n[3/3] Training SVM...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
models['SVM'] = {'model': svm, 'accuracy': svm_acc}
print(f"Accuracy: {svm_acc:.4f}")

# Select best
best_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
best_model = models[best_name]['model']
best_acc = models[best_name]['accuracy']

print("\n" + "="*60)
print(f"BEST MODEL: {best_name} ({best_acc:.4f})")
print("="*60)

# Show classification report
y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save
print("\nSaving model...")
joblib.dump(best_model, 'deepfake_audio_classifier.pkl')
joblib.dump(scaler, 'audio_scaler.pkl')

metadata = {
    'model_type': best_name,
    'accuracy': float(best_acc),
    'feature_extractor': 'facebook/wav2vec2-base-960h',
    'embedding_dim': int(X.shape[1]),
    'num_classes': int(len(np.unique(y))),
    'class_labels': {int(i): f'class_{i}' for i in np.unique(y)}
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Model saved!")
print("✓ Scaler saved!")
print("✓ Metadata saved!")