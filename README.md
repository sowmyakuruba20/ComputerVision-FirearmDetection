# Enhancing Firearm Detection in Video Surveillance

**Abstract**

This project aims to develop a robust, automated firearm detection system using deep learning to enhance real-time surveillance and improve public safety. We analyzed 398 videos from the Mendeley Data set using Faster R-CNN with ResNet-50 and YOLOv5 models. Data augmentation with SRGAN significantly boosted detection accuracy and speed. Faster R-CNN achieved an mAP of 0.995, and YOLOv5 achieved an mAP of 0.983. These results highlight the effectiveness of combining high-accuracy models with fast models, enhanced by SRGAN, for real-time firearm detection.

**Introduction**

The increase in concealed firearm incidents underscores the need for improved security measures. This study aims to leverage deep learning to create a robust, automated firearm detection system. We employed Faster R-CNN with ResNet-50 and YOLOv5 models, integrating SRGAN for data augmentation to enhance detection accuracy and speed.

**Methodology**

**Dataset Collection**
We used a dataset of 398 videos featuring individuals with and without firearms. The dataset includes categories for Handgun, Machine Gun, and No Gun.

**Data Pre-processing**
Preprocessing involved extracting frames at 25 FPS and 640x480 resolution, converting annotations to the required formats for YOLOv5 and Faster R-CNN, and applying SRGAN for image enhancement.

**Modeling**
Faster R-CNN with ResNet-50: This model includes a Region Proposal Network (RPN) and a Fast R-CNN detector. We modified the model to detect Machine Gun and Handgun.
YOLOv5: The model's architecture includes Backbone, Neck, and Head, designed for fast and accurate object detection. We tailored it to detect three specific classes: Machine Gun, Handgun, and No Gun.

**Results and Analysis**

Faster R-CNN: Achieved an mAP@0.5 of 99.5% and mAP@0.5:0.95 of 76.7% with GAN-augmented data.
YOLOv5: Achieved an mAP@0.5 of 0.983 and mAP@0.5:0.95 of 0.648 with GAN-augmented data.
Comparison:
Faster R-CNN: High accuracy but slower inference time.
YOLOv5: Faster inference but slightly lower accuracy.

**Conclusion**

Our project demonstrates the effectiveness of integrating high-accuracy and fast models with SRGAN for data augmentation in real-time firearm detection. Future research could focus on integrating real-time alert mechanisms and adaptive learning techniques to improve system performance and reliability.
