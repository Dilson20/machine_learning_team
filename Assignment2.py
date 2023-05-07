#%%
import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Set the directory containing the flower images
base_dir = 'Flowers'

# Define the names of the folders containing the different flower types
flower_names = ['Babi', 'Calimerio', 'Chrysanthemum', 'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']

# Define the size of the input images
img_size = (128, 128)

# Load the flower images and labels into arrays
images = []
labels = []
for i, flower_name in enumerate(flower_names):
    folder = os.path.join(base_dir, flower_name)
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        img = Image.open(file_path)
        img = img.resize(img_size)
        x = np.array(img)
        images.append(x)
        label = to_categorical(i, num_classes=len(flower_names))
        labels.append(label)
images = np.array(images)
labels = np.array(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the images
X_train = X_train / 255.0
X_val = X_val / 255.0

# Define the CNN architecture
input_shape = (img_size[0], img_size[1], 3)
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(flower_names), activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Extract the features from the trained model
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dense').output)
features = feature_extractor.predict(images)

# Define a function to recommend similar flower images
def recommend_similar_images(image_path, num_recommendations=10):
    # Load the input image
    img = Image.open(image_path)
    img = img.resize(img_size)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    # Extract the features from the input image
    features_x = feature_extractor.predict(x)
    
    # Compute the distances between the input image features and the features of all the images in the dataset
    distances = np.linalg.norm(features - features_x, axis=1)
    
    # Get the indices of the images with the smallest distances
    idxs = np.argsort(distances)[:num_recommendations]
    
    # Load and display the recommended images
    for i in idxs:
        folder_name = flower_names[np.argmax(labels[i])]
        file_name = os.listdir(os.path.join(base_dir, folder_name))[0]
        file_path = os.path.join(base_dir, folder_name, file_name)
        img = Image.open(file_path)
        img.show()
recommend_similar_images('Flowers/Babi/babi_1.jpg', 10)
#%%
import matplotlib.pyplot as plt
from PIL import Image

# open the image file
img = Image.open("Flowers/Flowers/Babi/babi_1.jpg")

# display the image
plt.imshow(img)
plt.show()
# %%
