## PyTorch Image Classification Models Comparison

This repository contains Python code for training and evaluating various PyTorch models for image classification tasks using the MNIST dataset. The code demonstrates the training and evaluation process for Convolutional Neural Networks (CNNs), Region-based CNN (R-CNN), and fine-tuning pre-trained models such as VGG16 and AlexNet.

### Requirements
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib

### Instructions
1. **Dataset Preparation**: The MNIST dataset will be automatically downloaded and preprocessed during the execution of the code. No additional setup is required.

2. **Model Training and Evaluation**:
   - The code provides implementations for training CNN, R-CNN, and fine-tuning VGG16 and AlexNet.
   - Each model's training process is displayed with loss updates during training and evaluated on the test set.
   - Accuracy, F1 Score, and Confusion Matrix are calculated for each model.

3. **Comparative Analysis**:
   - The accuracy of VGG16 and AlexNet after fine-tuning is compared.
   
### Code Explanation
- **`cnn_mnist`**: Defines and trains a simple Convolutional Neural Network (CNN) for MNIST classification.
- **`rcnn_mnist`**: Implements a Region-based CNN (R-CNN) architecture for MNIST classification.
- **`vgg16_alexnet_finetune`**: Fine-tunes pre-trained VGG16 and AlexNet models on the MNIST dataset.

### Usage
- Run each Python script individually to observe the training and evaluation process of the respective models.
- Make sure to have the necessary libraries installed (`torch`, `torchvision`, `scikit-learn`, `matplotlib`) before running the scripts.

### Acknowledgments
- The code is adapted from various PyTorch official documentation and tutorials.
- Pre-trained models are obtained from torchvision models.



### Understanding MyViT Model and Submission Generation

This code implements a custom Vision Transformer (ViT) model called `MyViT` for image classification tasks, and it demonstrates the process of generating submission-ready predictions on a test dataset. Additionally, it includes visualization of a few images with their true and predicted labels.

#### MyViT Model
- `MyViT` is a custom implementation of the Vision Transformer architecture for image classification.
- It consists of several components:
  1. **Linear Mapper**: Maps the input patches to a lower-dimensional space.
  2. **Classification Token**: Learnable token added to the input tokens for classification purposes.
  3. **Positional Embedding**: Embeddings added to represent the spatial position of each patch.
  4. **Transformer Encoder Blocks**: Stacked transformer blocks for feature extraction and representation.
  5. **MLP Classifier**: Multi-layer perceptron for final classification.
- The `forward` method processes input images through the model architecture and returns class predictions.

#### Submission Generation
- The code prepares submission-ready predictions for a test dataset.
- It iterates through the test dataset using a data loader, generates predictions using the trained model, and stores the predictions along with image IDs and true labels.
- The predictions are stored in a DataFrame for further analysis or submission to competitions.

#### Image Visualization
- The code also visualizes a few images along with their true and predicted labels.
- It iterates through a subset of the test dataset, generates predictions, and visualizes the images along with their labels.
- This visualization aids in understanding the model's performance visually.

### Instructions for Usage
1. **Model Training**: Train the `MyViT` model on your dataset or use pre-trained weights if available.
2. **Submission Generation**: Use the provided code to generate submission-ready predictions on your test dataset.
3. **Image Visualization**: Visualize a subset of images along with their true and predicted labels to understand model performance.

Ensure that your dataset is appropriately formatted and compatible with the model architecture before training and submission generation.

### References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision.models Documentation](https://pytorch.org/vision/stable/models.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

