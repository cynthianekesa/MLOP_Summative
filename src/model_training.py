import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Set the path to the dataset
DATASET_DIR = './data/data 1/DATASET/TEST' 

# Load the dataset
def load_data(dataset_dir):
    # Create a list of image file paths and their corresponding labels
    images = []
    labels = []
    
    # Iterate through each class directory
    for label, class_name in enumerate(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                images.append(img_path)
                labels.append(label)  # 0 for 'O', 1 for 'R'
    
    return images, labels

# Load images and labels
image_paths, labels = load_data(DATASET_DIR)

# Function to load and preprocess images
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Decode JPEG images
    img = tf.image.resize(img, [180, 180])  # Resize to 180x180
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Create a TensorFlow dataset
def create_dataset(image_paths, labels):
    # Create a TensorFlow Dataset from the image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))  # Preprocess images
    dataset = dataset.shuffle(buffer_size=len(image_paths))  # Shuffle the dataset
    dataset = dataset.batch(32)  # Set batch size
    return dataset

# Create the dataset
train_dataset = create_dataset(image_paths, labels)

# Clear the session to avoid variable conflicts
tf.keras.backend.clear_session()

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(180, 180, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)  # Assuming 2 classes: 'O' and 'R'
])

# Create a new optimizer instance
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Save the model
model.save('my_model.h5')  # Save the model to a file