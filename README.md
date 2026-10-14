# MNIST Handwritten Digit Classification Using CNN

A deep learning project that uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to classify handwritten digit images from **0 to 9** using the MNIST dataset.

The project covers image preprocessing, CNN model training, evaluation, and visualization of model performance.

## Features

- Loads the MNIST dataset through Keras.
- Normalizes pixel values to the range `0–1`.
- Reshapes images to the CNN input format `(28, 28, 1)`.
- Visualizes sample training images.
- Trains a CNN with convolution, max-pooling, dense, and dropout layers.
- Evaluates the model on the test dataset.
- Plots training and validation accuracy and loss.
- Generates a confusion matrix.
- Displays sample predictions alongside their actual labels.
- Saves the trained model as `mnist_cnn.keras`.

## Dataset

The project uses the MNIST handwritten digit dataset, downloaded automatically by Keras when the script or notebook is run.

- **Image dimensions:** `28 × 28` pixels
- **Image format:** Grayscale
- **Number of classes:** 10
- **Labels:** Digits `0–9`

## Model Architecture

The CNN uses the following layers:

```text
Input: 28 × 28 × 1
       ↓
Conv2D (32 filters, 3 × 3, ReLU)
       ↓
MaxPooling2D (2 × 2)
       ↓
Conv2D (64 filters, 3 × 3, ReLU)
       ↓
MaxPooling2D (2 × 2)
       ↓
Flatten
       ↓
Dense (128 units, ReLU)
       ↓
Dropout (30%)
       ↓
Dense (10 units, Softmax)
```

The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as its evaluation metric. The training script runs for **5 epochs** with a batch size of **64** and reserves 10% of the training data for validation.

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Structure

```text
mnist-cnn-classification/
├── README.md
├── mnist_cnn.ipynb
├── train.py
├── requirements.txt
└── .gitignore
```

- `train.py` — preprocessing, model training, evaluation, plotting, prediction visualization, and model saving.
- `mnist_cnn.ipynb` — notebook version of the project workflow.
- `requirements.txt` — Python package dependencies.
- `.gitignore` — excludes generated model files and common local development artifacts.

## Getting Started

### Prerequisites

- Python installed
- `pip` for package installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your Git repository URL and local folder name.

### 2. (Recommended) Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training script

```bash
python train.py
```

The script downloads the MNIST dataset if needed, trains the CNN, prints the test loss and accuracy, displays evaluation plots, and saves the trained model to `mnist_cnn.keras`.

### Run the notebook instead

Start Jupyter:

```bash
jupyter notebook
```

Open `mnist_cnn.ipynb` and run its cells in order.

## Outputs

After running the training script, you should see:

- Sample MNIST training images
- Training vs. validation accuracy plot
- Training vs. validation loss plot
- Test loss and test accuracy in the terminal
- Confusion matrix for the ten digit classes
- Sample predictions showing actual and predicted labels
- Saved model file: `mnist_cnn.keras`

The actual test accuracy is printed when the model runs; it can vary with the environment and training settings. The saved model is excluded from version control by the project's `.gitignore`.

## Skills Demonstrated

- Convolutional Neural Networks (CNNs)
- Image preprocessing and normalization
- Multiclass image classification
- Model training and evaluation
- Dropout for regularization
- Training and validation curve analysis
- Confusion matrix interpretation
- TensorFlow/Keras model serialization

## Possible Improvements

- Add data augmentation.
- Build an interactive digit-drawing interface.
- Compare different CNN architectures and hyperparameters.
- Add a separate inference script to load the saved model and classify new images.

## Author

Vamsi Krishna
