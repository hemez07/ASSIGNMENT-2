
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



from sklearn.linear_model import LogisticRegression

# Train Softmax classifier (Logistic Regression)
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax.fit(X_train_pca, y_train)

# Predict and evaluate Softmax
y_pred_softmax = softmax.predict(X_test_pca)
softmax_accuracy = accuracy_score(y_test, y_pred_softmax)
print(f"Softmax Accuracy with PCA: {softmax_accuracy:.4f}")




import torch.nn as nn
import torch.optim as optim

# Define Two-Layer Neural Network
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # <-- Fixed __init__
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # No softmax needed as CrossEntropyLoss applies it internally

# Define parameters
input_size = 100  # PCA-reduced features
hidden_size = 100
output_size = len(selected_classes)

# Initialize model, loss function, and optimizer
model = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Train neural network
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate neural network
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_predictions = torch.argmax(test_outputs, dim=1)
    nn_accuracy = accuracy_score(y_test_tensor.numpy(), test_predictions.numpy())

print(f"Neural Network Accuracy with PCA: {nn_accuracy:.4f}")




# Compare classifier accuracies
print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Softmax Accuracy: {softmax_accuracy:.4f}")

# Plot performance comparison
plt.bar(["SVM", "Softmax"], [svm_accuracy, softmax_accuracy])
plt.ylabel("Accuracy")
plt.title("Classifier Comparison")
plt.show()