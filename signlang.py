import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load the train dataset
train_csv_path = "E:\\sign_mnist_train.csv"
df_train = pd.read_csv(train_csv_path)

# Load the test dataset
test_csv_path = "E:\\sign_mnist_test.csv"
df_test = pd.read_csv(test_csv_path)

# Filter out instances with label values of 24
valid_labels_mask_train = df_train['label'] < 24
df_train = df_train[valid_labels_mask_train]

# Split the data into training and test sets
df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42)

# Extract labels and pixel values
labels_train = df_train['label'].values
pixels_train = df_train.iloc[:, 1:].values.reshape(-1, 28, 28, 1)

labels_test = df_test['label'].values
pixels_test = df_test.iloc[:, 1:].values.reshape(-1, 28, 28, 1)

# Print unique labels after correction
unique_labels_train = np.unique(labels_train)
unique_labels_test = np.unique(labels_test)
print(f'Unique labels in training set after correction: {unique_labels_train}')
print(f'Unique labels in test set after correction: {unique_labels_test}')

# Convert labels to one-hot encoding
encoder = LabelEncoder()
labels_train_encoded = encoder.fit_transform(labels_train)
labels_test_encoded = encoder.transform(labels_test)

labels_train_one_hot = to_categorical(labels_train_encoded)
labels_test_one_hot = to_categorical(labels_test_encoded)

# Normalize pixel values to the range [0, 1]
pixels_train = pixels_train / 255.0
pixels_test = pixels_test / 255.0

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(23, activation='softmax'))  # 25 classes (0-23, excluding 9 and 24)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(pixels_train, labels_train_one_hot, epochs=10, validation_data=(pixels_test, labels_test_one_hot))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(pixels_test, labels_test_one_hot)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save("sign_language_model.h5")
