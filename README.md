# PRODIGY MACHINE LEARNING TASK : 03

# PROBLEM STATEMENT:
# Cat and Dog Classification using Support Vector Machine (SVM)

As part of my delightful internship experience with Prodigy Infotech, I am excited to present Task 3: implementing a Support Vector Machine (SVM) to classify images of cats and dogs. The dataset used for this project is sourced from Kaggle. The objective is to build a model that can accurately classify whether an image contains a cat or a dog.

## Introduction

In this project, we use a Support Vector Machine (SVM), a supervised learning model, to classify images of cats and dogs. SVMs are effective in high-dimensional spaces and are memory efficient. This project demonstrates the steps involved in data preprocessing, model training, and evaluation using SVM.

## Dataset

The dataset used in this project is the Cats and Dogs dataset from Kaggle. It contains thousands of images of cats and dogs, which are split into training and testing sets.

## Data Preprocessing

1. **Loading Data**:
    - Load images from the dataset directory. The images are stored in subdirectories named `cat` and `dog` under both `train` and `test` directories.
    - Use a library like OpenCV to read the images. Ensure that all images are loaded correctly and handle any corrupted images.

2. **Resizing Images**:
    - Resize all images to a fixed size (e.g., 64x64 pixels). This ensures uniformity in the input data and reduces computational complexity. You can use OpenCV's `cv2.resize` function for this purpose.

3. **Grayscale Conversion (Optional)**:
    - Convert images to grayscale if you plan to use a linear SVM or if you want to reduce the dimensionality of the input data. This can be done using OpenCV's `cv2.cvtColor` function.

4. **Feature Extraction**:
    - Flatten each image into a one-dimensional feature vector. For instance, a 64x64 RGB image will be flattened into a vector of length 64 * 64 * 3 = 12,288. If using grayscale, the vector length will be 64 * 64 = 4,096.

5. **Normalization**:
    - Normalize pixel values to the range [0, 1] by dividing by 255. This helps in faster convergence during training.

6. **Label Encoding**:
    - Encode the labels (cat and dog) into binary values. For instance, `cat` can be encoded as 0 and `dog` as 1. This can be done using scikit-learn's `LabelEncoder`.


## Model Training

1. **Splitting Data**:
    - Split the preprocessed data into training and validation sets. A common split is 80% training and 20% validation. Use scikit-learn's `train_test_split` function for this purpose.

2. **Initializing SVM**:
    - Initialize the SVM model with a suitable kernel. The choice of kernel (e.g., linear, RBF) depends on the nature of the data and the problem. For this project, we start with a linear kernel and then experiment with RBF.

3. **Training SVM**:
    - Train the SVM model on the training data using the `fit` method. This involves finding the optimal hyperplane that separates the classes (cats and dogs).

4. **Hyperparameter Tuning**:
    - Perform hyperparameter tuning to find the best parameters for the SVM. This can be done using grid search or cross-validation techniques provided by scikit-learn's `GridSearchCV`.


## Model Evaluation

1. **Evaluating Performance**:
    - Evaluate the trained model on the validation set using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insights into the model's performance.
    - Use scikit-learn's `classification_report` and `confusion_matrix` to get detailed evaluation metrics.

2. **Plotting Confusion Matrix**:
    - Plot a confusion matrix to visualize the model's performance in distinguishing between cats and dogs. This helps identify any misclassification patterns.

3. **Visualizing Results**:
    - Visualize some example predictions to qualitatively assess the model's performance. Display a few test images along with their predicted labels and true labels.


## Results

- Report the final performance metrics of the model, including accuracy, precision, recall, and F1-score.
- Include some example images with their predicted labels to provide a visual representation of the model's performance.
- Discuss potential improvements and future work, such as experimenting with different kernels, data augmentation, or using deep learning models.

Any Question ??
email : magnel2001@gamil.com
