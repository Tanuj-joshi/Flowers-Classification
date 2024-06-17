# Flowers-Classification-using-Pytorch
### An Image Classifier to recognize different species of Flowers

![Description of image](intro-image.png)

This repository contains the code and resources to train a Convolutional Neural Network (CNN) to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. We'll be using the dataset of 102 flower categories.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset

The full dataset can't be uploaded here in Github due to space limitation. The full dataset of 102 flower categories can be obtained here, https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. Here, we have divided the dataset into training, validation and test sets to evaluate the performance of the model.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.7 or higher
- Torch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (optional, for interactive experimentation)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Tanuj-joshi/Flowers-Classification-using-Pytorch.git
   cd Flowers-Classification-using-Pytorch
   ```

2. Create a virtual environment and activate it::

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from given link and place it in the Dataset/ directory.

## Usage

### Training the Model

1. Ensure that the dataset is properly placed in the Dataset/ directory as 'train', 'valid' and 'test' folders.

2. Run the training script:

   ```bash
   python train.py --image_dir Dataset/ --arch < vgg13/densenet121 > --epoch 50 --batch_size 16 --model_path models/
   ```
   This script will trains the vgg13/densenet121 (pretrained) model on the training dataset and save the trained model to the models/ directory.

### Evaluation

 To evaluate the performance of the trained model on the test set, run:

   ```bash
    python evaluate.py --image_dir Dataset/test/ --arch vgg13 --batch_size 16 --model_path models/epoch3_classifier.pt
   ```
 This script will load the trained model and output the Accuracy, Loss, Precision, recall and F1 score on the test set.

 To predict output of the trained model on a single image, run:

   ```bash
    python evaluate.py --image_dir Dataset/< image path > --arch vgg13 --batch_size 16 --model_path models/epoch3_classifier.pt
   ```
 This script will load the trained model and display the image of the flower with its predicted class name.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

Special thanks to the PyTorch teams for their excellent deep learning frameworks.
