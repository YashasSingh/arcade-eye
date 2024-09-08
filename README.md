
## Eye Tracking Model using TensorFlow and ResNet
![image](https://github.com/user-attachments/assets/32383b05-20fe-4bbd-b307-298f955eb83a)


This project demonstrates a simple implementation of an eye-tracking model using TensorFlow, leveraging the ResNet152V2 model for feature extraction. The model is trained to predict the coordinates of the left and right eyes based on input images. 

### Requirements

Before running the notebook, ensure the following packages are installed:

```bash
!pip install labelme tensorflow opencv-python matplotlib albumentations
```

### Notebook Structure

1. **Imports & Setup:**
   - Libraries like TensorFlow, OpenCV, Matplotlib, and Albumentations are used.
   - ResNet152V2 is used for the neural network's backbone.

2. **Loading Images:**
   - Images are loaded using TensorFlow's `tf.data.Dataset.list_files` and mapped through functions to resize and normalize.
   - Train, test, and validation datasets are prepared from image and label paths.

3. **Label Loading:**
   - Labels are JSON files that contain the coordinates of the eyes.
   - A custom function is used to extract keypoints from these files.

4. **Data Preprocessing:**
   - Images are resized to 250x250 and normalized by dividing pixel values by 255.
   - Images and labels are zipped into a TensorFlow dataset, which is batched and shuffled.

5. **Network Architecture:**
   - A Sequential model is defined using:
     - ResNet152V2 as the backbone.
     - Convolutional layers for feature extraction.
     - Dropout layers for regularization.
     - The output layer predicts 4 coordinates representing the eye positions.

6. **Training the Model:**
   - The model is trained with `MeanSquaredError` as the loss function and Adam optimizer.
   - Training is done for 100 epochs with a validation set for monitoring performance.

7. **Model Evaluation:**
   - The model's performance is visualized by plotting training and validation loss curves.
   - Predictions are made on test images, and eye locations are drawn on images using OpenCV's `circle` function.

8. **Model Saving & Loading:**
   - The model is saved as `eyetrackerresnet.h5` and can be reloaded for future use.

9. **Real-Time Eye Tracking:**
   - A webcam feed is used to capture live images.
   - The model makes predictions on the live frames, and the coordinates of the eyes are displayed in real time.

### Instructions for Use

1. **Training the Model:**
   - To train the model, simply run the notebook from start to finish. Ensure that the dataset is properly structured with `train`, `test`, and `val` directories containing images and their corresponding JSON labels.

2. **Running Real-Time Tracking:**
   - After training the model, you can use the webcam to track eyes in real-time by executing the last section of the notebook. Press 'q' to quit the webcam feed.

### Customization

- You can adjust the model architecture by modifying the layers in the `Sequential` model.
- Change the learning rate, batch size, or number of epochs for different training results.

### Dataset Structure

```bash
aug_data/
├── train/
│   ├── images/
│   ├── labels/
├── test/
│   ├── images/
│   ├── labels/
├── val/
│   ├── images/
│   ├── labels/
```

### Important Notes

- Make sure to have a proper setup for your dataset.
- The model expects images of size 250x250 pixels. If your images are larger or smaller, adjust accordingly.
  
