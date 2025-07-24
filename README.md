# PinchBot: Long-Horizon Deformable Manipulation with Guided Diffusion Policy
[[arXiv]](TODO) [[Project Website]](https://sites.google.com/andrew.cmu.edu/pinchbot/home) [Demonstration Dataset](https://drive.google.com/drive/folders/13s9nUNslSc55CU0rR_AVe5hVQML0wLCm)

Pottery creation is a complicated art form that requires dexterous, precise and delicate actions to slowly morph a block of clay to a meaningful, and often useful 3D goal shape. In this work, we aim to create a robotic system that can create simple pottery goals with only pinch-based actions. This pinch pottery task allows us to explore the challenges of a highly multi-modal and long-horizon deformable manipulation task. To this end, we present PinchBot, a goal-conditioned diffusion policy model that when combined with pre-trained 3D point cloud embeddings, task progress prediction and collision-constrained action projection, is able to successfully create a variety of simple pottery goals.

## Download Dataset
Follow the link to the [Demonstration Dataset](https://drive.google.com/file/d/1QN1vTGvsCvwakqlCOFcbvscUXZoP7eVo/view), download and unzip the files and update the path in scripts

## Setup PointBERT
Follow the installation instructions and download the model weights of [Point-BERT](https://github.com/Julie-tang00/Point-BERT)

## Train Policies
To train a point cloud-based sculpting policy, run train_policy.py. Update relevant parameters for choice of point cloud embedding and guidance scenarios. 

## Replicate Harware Setup
Follow the link to the [Hardware CAD](https://drive.google.com/file/d/1JbbHU8lW7LBvTGYZ2qOLVUAGpOO5gcLK/view?usp=drive_link) to replicate our camera cage.