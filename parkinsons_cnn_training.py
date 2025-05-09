import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# Define image size
image_size = (224, 224)  # MobileNet's input size
class_names = ['normal', 'parkinson']  # Adjust based on your classes

# Preprocess the images
def preprocess_images(images):
    images_resized = [cv2.resize(img, image_size) for img in images]
    images_resized = np.array(images_resized)
    images_resized = images_resized / 255.0  # Normalize to [0, 1]
    return images_resized

# Convert labels to one-hot encoding
def prepare_labels(labels, num_classes=2):
    return to_categorical(labels, num_classes=num_classes)

# Load and prepare dataset
class DataSet:
    def __init__(self, path, categories, lheight, lwidth, grayscale, count, shuffled, multiclass):
        self.path = path
        self.categories = categories
        self.lheight = lheight
        self.lwidth = lwidth
        self.grayscale = grayscale
        self.count = count
        self.shuffled = shuffled
        self.multiclass = multiclass
        self.dataset = self.load_data()

    def load_data(self):
        images = []
        labels = []

        for label in self.categories:
            folder = os.path.join(self.path, label)
            for filename in os.listdir(folder)[:self.count]:
                img = cv2.imread(os.path.join(folder, filename))
                if self.grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_resized = cv2.resize(img, (self.lwidth, self.lheight))
                images.append(img_resized)
                labels.append(self.categories.index(label))

        if self.shuffled:
            temp = list(zip(images, labels))
            np.random.shuffle(temp)
            images, labels = zip(*temp)

        return np.array(images), np.array(labels)

# Path to your dataset folder
dataset_path = "parkinsons_dataset"  # Adjust this to your dataset path
dataset = DataSet(dataset_path, categories=class_names, lheight=224, lwidth=224, grayscale=False, count=1000, shuffled=True, multiclass=True)

train_images, train_labels = dataset.dataset

# Preprocess images and labels
train_images = preprocess_images(train_images)
train_labels = prepare_labels(train_labels, num_classes=len(class_names))

# Split data into training and validation sets (optional)
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.06, random_state=42)

# Define the MobileNet model for multi-class classification
def build_mobilenet_model():
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    mobilenet.trainable = False  # Freeze the base model layers
    model = Sequential()
    model.add(mobilenet)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Output layer for multi-class classification
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and compile the model
model = build_mobilenet_model()

# Set up callbacks
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
es = EarlyStopping(monitor='val_loss', patience=5)

# Training parameters
batch_size = 16
epochs = 50

# Train the model
history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback, es]
)

# Clear session after training to free up GPU memory
K.clear_session()

# Save the trained model (optional)
model.save('parkinsons_model.h5')

# Optionally: Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

