# Image Classification with VGG16
This project uses the pre-trained VGG16 model for image classification. The model is fine-tuned with specific data to predict four classes of images.

## Code Overview
The code performs the following steps:

### 1. Load the Pre-trained Model: 
VGG16 without the final classification layer (include_top=False).
### 2. Add Custom Layers:
- Flatten the output of the VGG16 model.
- Add a dense layer with 512 neurons and ReLU activation.
- Add a final dense layer with softmax activation for classifying into 4 classes.
### 3. Compile the Model: 
Uses Adam optimizer and categorical_crossentropy loss function.
### 4. Data Preprocessing:
- Uses ImageDataGenerator to resize and normalize images.
- Loads training, validation, and test sets from specific directories.
### 5. Train the Model: 
Trains for 25 epochs with training and validation data.
### 6. Evaluate the Model:
Evaluates the model on the test set and prints accuracy.
### 7. Visualization:
Plots training and validation accuracy and loss over epochs.

## Instructions
### 1. Prerequisites: Ensure you have the following libraries installed:

TensorFlow
Scikit-image
Matplotlib
### 2. Prepare the Data: 
Organize your images into train, test, and val directories with subdirectories for each class.

### 3. Run the Script: 
Execute the script to train the model and visualize accuracy and loss curves.
