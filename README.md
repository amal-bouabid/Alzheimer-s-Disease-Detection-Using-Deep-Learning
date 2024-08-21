<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Classification with VGG16</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
</head>
<body>

    <h1>Image Classification with VGG16</h1>

    <p>This project uses the pre-trained VGG16 model for image classification. The model is fine-tuned with specific data to predict four classes of images.</p>

    <h2>Code Overview</h2>

    <p>The code performs the following steps:</p>
    <ol>
        <li><strong>Load the Pre-trained Model</strong>: VGG16 without the final classification layer (<code>include_top=False</code>).</li>
        <li><strong>Add Custom Layers</strong>:
            <ul>
                <li>Flatten the output of the VGG16 model.</li>
                <li>Add a dense layer with 512 neurons and ReLU activation.</li>
                <li>Add a final dense layer with softmax activation for classifying into 4 classes.</li>
            </ul>
        </li>
        <li><strong>Compile the Model</strong>: Uses Adam optimizer and <code>categorical_crossentropy</code> loss function.</li>
        <li><strong>Data Preprocessing</strong>:
            <ul>
                <li>Uses <code>ImageDataGenerator</code> to resize and normalize images.</li>
                <li>Loads training, validation, and test sets from specific directories.</li>
            </ul>
        </li>
        <li><strong>Train the Model</strong>: Trains for 25 epochs with training and validation data.</li>
        <li><strong>Evaluate the Model</strong>: Evaluates the model on the test set and prints accuracy.</li>
        <li><strong>Visualization</strong>:
            <ul>
                <li>Plots training and validation accuracy and loss over epochs.</li>
            </ul>
        </li>
    </ol>

    <h2>Instructions</h2>

    <ol>
        <li><strong>Prerequisites</strong>: Ensure you have the following libraries installed:
            <ul>
                <li>TensorFlow</li>
                <li>Scikit-image</li>
                <li>Matplotlib</li>
            </ul>
        </li>
        <li><strong>Prepare the Data</strong>: Organize your images into <code>train</code>, <code>test</code>, and <code>val</code> directories with subdirectories for each class.</li>
        <li><strong>Run the Script</strong>: Execute the script to train the model and visualize accuracy and loss curves.</li>
    </ol>

    <h2>Code</h2>

    <pre><code>
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configuration
input_size = (224, 224)
num_classes = 4

# Create VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_size[0], input_size[1], 3))
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/content/drive/MyDrive/train', target_size=input_size, batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('/content/drive/MyDrive/test', target_size=input_size, batch_size=32, class_mode='categorical', shuffle=False)
valid_generator = valid_datagen.flow_from_directory('/content/drive/MyDrive/val', target_size=input_size, batch_size=32, class_mode='categorical')

# Train the model
history_VGG16 = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=25, validation_data=valid_generator, validation_steps=len(valid_generator))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', test_acc)

# Visualization
acc = history_VGG16.history['accuracy']
val_acc = history_VGG16.history['val_accuracy']
loss = history_VGG16.history['loss']
val_loss = history_VGG16.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, '-b', label='accuracy')
plt.plot(epochs, val_acc, '--r', label='val_acc')
plt.legend()
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, '-b', label='loss')
plt.plot(epochs, val_loss, '--r', label='val_loss')
plt.legend()
plt.title('Training and validation loss')
plt.show()
    </code></pre>

</body>
</html>
