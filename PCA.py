import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")

# Dataset path and image settings
dataset_path = "DataSet"
image_size = (128, 256)  # (Height, Width)

# Load images and labels
image_data = []
labels = []
image_extensions = {".jpg", ".jpeg", ".png"}

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            ext = os.path.splitext(img_name)[-1].lower()
            if ext in image_extensions:
                try:
                    img = Image.open(img_path).convert("L").resize(image_size, Image.LANCZOS)
                    image_data.append(np.array(img).flatten())
                    labels.append(person_name)
                except Exception as e:
                    print(f"❌ Skipping {img_path} due to error: {e}")

#  Convert to numpy arrays
X = np.array(image_data)
y_names = np.array(labels)

# Encode person names to numbers
le = LabelEncoder()
y = le.fit_transform(y_names)
target_names = le.classes_

# Choose which image index to display
index = 2  #Change this to any valid index

if index < len(X):
    #  Display an original image
    plt.figure(figsize=(6, 3))
    plt.imshow(X[index].reshape(image_size), cmap='gray')
    plt.title(f"Original Image: {target_names[y[index]]}")
    plt.axis('off')
    plt.show()
else:
    print(f"Index {index} is out of range. Dataset has only {len(X)} images.")

#  PCA (dimensionality reduction)
n_components = min(6, len(X))  # Limit PCA to number of samples
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

# Reconstruct image using PCA
X_reconstructed = pca.inverse_transform(X_pca)

if index < len(X):
    plt.figure(figsize=(6, 3))
    plt.imshow(X_reconstructed[index].reshape(image_size), cmap='gray')
    plt.title("PCA-Reconstructed Image")
    plt.axis('off')
    plt.show()

# Train SVM on the same data (no test split due to small size)
clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
clf.fit(X_pca, y)
y_pred = clf.predict(X_pca)

# Accuracy
print("✅ Accuracy:", clf.score(X_pca, y))

# 📊 Evaluation (on same data)
print("\n🔍 Classification Report:\n")
print(classification_report(y, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

# Optional: Show all original images
# for i in range(len(X)):
#     plt.imshow(X[i].reshape(image_size), cmap='gray')
#     plt.title(target_names[y[i]])
#     plt.axis('off')
#     plt.show()
