
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define dataset transformation (convert to tensors)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define labels subset (3 classes)
selected_classes = {0, 1, 2}  # Example: Airplane, Automobile, Bird

# Filter dataset using NumPy
train_mask = np.isin(trainset.targets, list(selected_classes))
test_mask = np.isin(testset.targets, list(selected_classes))

X_train, y_train = trainset.data[train_mask], np.array(trainset.targets)[train_mask]
X_test, y_test = testset.data[test_mask], np.array(testset.targets)[test_mask]

# Reshape images to (num_samples, 3072) and normalize
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0  

# Apply PCA (reduce to 100 components)
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# # Display explained variance ratio
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Explained Variance')
# plt.title('PCA - Explained Variance vs Components')
# plt.show()

# Select a random index from the filtered dataset
random_idx = np.random.randint(len(X_train))

# Reshape the image back to (32, 32, 3) format
sample_image = X_train[random_idx].reshape(32, 32, 3)

# Display the image
plt.imshow(sample_image)
plt.title(f"Sample Image - Class {y_train[random_idx]}")
plt.axis("off")
plt.show()



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train_pca, y_train)

# Predict and evaluate SVM
y_pred_svm = svm.predict(X_test_pca)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy with PCA: {svm_accuracy:.4f}")
