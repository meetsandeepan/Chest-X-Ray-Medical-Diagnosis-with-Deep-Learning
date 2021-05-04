#weekendproject
1. Chest X-Ray Medical Diagnosis with Deep Learning
Hi all, I am creating this series to fulfil 2 purposes, primarily to document things I do to use it as a personal archive (those who know me will tell you, i have a habit of forgetting things) next, to create a writeup of steps I've taken to reach the goal.( Math is boring, ik, hence i will try to omit most of 'em)
As part of a course that I'm doing I am taking a public ChestX-ray8 dataset (https://arxiv.org/abs/1705.02315) which has 14 labels of pathological conditions (eg: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion) and using DenseNet121 (https://arxiv.org/pdf/1608.06993) to create a classifier to give binary output for each of the classes.
Step 1: Pre-process and prepare a real-world X-ray dataset
Each image in the data set contains multiple text-mined labels identifying 14 different pathological conditions. These in turn can be used by physicians to diagnose 8 different diseases. I will use this data to develop a single model that will provide binary classification predictions for each of the 14 labeled pathologies. In other words it will predict 'positive' or 'negative' for each of the pathologies. Out of these images ~62% were used for training the model, ~7% to validate the model & ~30% for testing the model.
One of the things to look out for in this step is data leakage. This happens when a patient goes to hospital and takes different Chest-XRays at different times which might exist in across our sets, your model should not have any information about the test set.
Preparing images
1. Normalize the mean and standard deviation of the data
2. Shuffle the input after each epoch.
3. Set the image size to be 320px by 320px
Make sure to use different generator for test and valid sets.
Step 2: Use transfer learning to retrain a DenseNet model for X-ray image classification
This is a technique commonly used for say if we are to identify Chest-XRays we train the model to identify human faces first then transfer the weights learned to identify Chest-XRays the idea behind is if the model can identify edges in faces that might also help in segregating parts of the X-Rays.
Step 3: Learn a technique to handle class imbalance
One of the challenges with working with medical diagnostic datasets is the large class imbalance present in such datasets. The positive labels( where the the patient has the condition) to negative labels( where the patient doesn't have the condition) ration is really low. I am plotting attaching a plot to show a positive to negative labels for each class in the set. Mind here we have taken ~1200 images for easy manipulation.
One way to overcome this is to add weights to the the labels by that our cumulative loss function can be reduced. After computing the weights, our final weighted loss for each training case will be sum(cross-entropy) = -(wp* y*log(f(x))+wn*(1−y)log(1−f(x))).
Step 4: Measure diagnostic performance by computing the AUC (Area Under the Curve) for the ROC (Receiver Operating Characteristic) curve
Here a pre-trained DenseNet121 model is used which we can load directly from Keras and then add two layers on top of it, backend Tensorflow.
1. A GlobalAveragePooling2D layer to get the average of the last convolution layers from DenseNet121.
2. A Dense layer with sigmoid activation to get the prediction logits for each of our classes.
After training (which took about 4 hours to complete using Nvidia K80 gpu in AWS, it has about ~24GB of memory and our dataset size is ~40GB so a batch size of 32 is taken) we can evaluate our test set. Here is the AUC curve is attached for our model. You can compare the performance to the AUCs reported in the original ChexNeXt paper in the table below. For reference, here's the AUC figure from the ChexNeXt paper which includes AUC values for their model as well as radiologists on this dataset.
For details about the best performing methods and their performance on this dataset, we encourage you to read the following papers
CheXNet(https://arxiv.org/abs/1711.05225)
CheXpert(https://arxiv.org/pdf/1901.07031.pdf)
ChexNeXt(https://journals.plos.org/plosmedicine/article...)
Step 5: Visualize model activity using GradCAMs
One of the most common approaches aimed at increasing the interpretability of models for computer vision tasks is to use Class Activation Maps (CAM). Class activation maps are useful for understanding where the model is "looking" when classifying an image. Here a GradCAM technique is used to produce a heatmap highlighting the important regions in the image for predicting the pathological condition. Now let's look at a few specific images. 4 of them are attached below (fun part, pheww)
Thank you for reading, if it peaks anyone's interest and someone wants to try it hands on I will include all of the codes in my github.