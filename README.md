# Udacity-MLND-Capstone
Capstone project for ML nanodegree

## Domain Background

Improving the quality of degradated images has a wide spectrum of applications like recovery of lower quality images transmitted over the network in-situ on ultra high defnition monitors equipped with computing devices capable to perform the recovery, or improving the quality of images taken by security systems.

## Problem Statement
As photographers, sometimes we get disappointed when we view our photos or video footage on a large screen. An image that we were convinced was crisp isn't so sharp when enlarged. To solve this, some people are using machine learning to re-create the image to add back the information lost by the blurr. There are applications available to correct photos using machine learning but since these applications were launch, significant progress have been made in the field of image restoration. 

## Datasets and Inputs
The inputs to our model will be images taken from blurry video and blurry photos. They should be resized to a standardized size and if need be a size that we will be able to transfer easily to our endpoint. 

## Solution Statement
We propose to use a state-of-the-art model to de-blurr photographs, and possibly videos. If time allows, the same approach will be used to remove noise from the same images as well. 

## Benchmark Model
The benchmark will be the demo script in https://github.com/swz30/MPRNet on the same image, and ensuring that we can obtain similar variance of the Laplacian. This will be compared with the result we are getting from the web app.

## Evaluation Metrics
We will quantify blurr with the variance of a Laplacian as explained in https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/ The results of implementing the web-app will be compared between original images and de-blurred ones.

## Project Design
![alt text](https://camo.githubusercontent.com/243ed29141814cacb56499e03fb9cda6eb3f709e3511d7b96c1847273e23fa57/68747470733a2f2f692e696d6775722e636f6d2f363963307051762e706e67 "MPRNet")
The model we use is depicted in the figure above. It is implemented in PyTorch and what we propose to do is to deploy this model to a Web Application, allowing a user to input a photo or perhaps a video and improve its quality.

