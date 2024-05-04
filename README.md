# ec523-lip-reading-project
Group Project for EC523 Deep Leaning course

Usman Jalil, Chris Krenz, Cole Resurreccion, Thomas Simmons
{ujalil,ckrenz,coler,tsimmons}@bu.edu

This TensorFlow project implements a 3D CNN to perform lip-reading given a video input.


## Overview

For this project, we are utilizing 2 datasets (LRW and LipNET) discussed further in our submitted report.

(Completed) We start by pre-processing the data: using a Haar classifier (unnecessary for LRW) to locate the speaker's lips, crop the image, turn it to grayscale, standardize its size, and normalize it.  For LipNET, this results in clips that are 75 frames long x 80 pixels wide x 40 pixels tall, and for LRW, it results in 29 frames of roughly 80 x 130.

(Partially Completed) After pre-processing, we then create a CNN with the following architecture:
model = models.Sequential(
    layers.Conv3D(32,   (3, 3, 3), activation='relu', padding='same', input_shape=(num_frames, height, width, channels)),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(64,   (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((2, 2, 2)),
    layers.Conv3D(128,  (3, 3, 3), activation='relu', padding='same'),
    layers.MaxPooling3D((1, 2, 2)), 
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
)

(Forthcoming) Finally, we train the model, test it, and visualize the results.

Additional information can be found in our submitted pdf.

## Data

The datasetse are discussed in the following sources:
  - J. S. Chung and A. Zisserman, “Lip Reading in the Wild,” in Computer Vision –  ACCV 2016, S.-H. Lai, V. Lepetit, K. Nishino, and Y. Sato, Eds., in Lecture Notes in Computer Science. Cham: Springer International Publishing, 2017, pp. 87–103. doi: 10.1007/978-3-319-54184-6_6.
  - J. S. Chung, A. Senior, O. Vinyals, and A. Zisserman, “Lip Reading Sentences in the Wild,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul. 2017, pp. 3444–3453. doi: 10.1109/CVPR.2017.367.
  - Y. M. Assael, B. Shillingford, S. Whiteson, and N. de Freitas, “LipNet: End-to-End Sentence-level Lipreading.” arXiv, Dec. 16, 2016. doi: 10.48550/arXiv.1611.01599.


# File Instructions

To run the App:
- cd App
- python3 app.py
