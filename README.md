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

</body>
</html>

