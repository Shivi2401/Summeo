# Summeo
Summeo is an AI video Summarizer currently in development phase.
We are following three different approaches for video summarization with SumMe dataset by removing the repetitive frames and giving importance to the distinct frames in the video. First approach is to generate self-similarity matrix of the video using the features extracted from pretrained imagenet models and to build a simple multi-layer perceptron for multi-label classification. Next approach is using 2D Convolution for extracting features from self-similarity matrix and fully connected layers for further classification. Last approach is the extension of the previous approach by using Particle Swarm Optimization method for hyper parameter tuning.

## Workflow of Summeo
<p align="center" ><img src="Summeo Workflow.png"></p>

## Applications
Some significant applications of video summarisation in the contemporary world or in the future are creating highlights of sports videos, trailer generation from the inputted movie, and even for many security purposes like the movement of a thief caught in closed circuit cameras during the night which is an abnormal situation included in the summarised video.
