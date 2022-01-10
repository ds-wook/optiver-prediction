# Optiver-prediction
### Introduction
This repository is the code that [Optiver Realized Volatility Prediction](https://www.kaggle.com/c/optiver-realized-volatility-prediction) competition.
### Learning Process
[Learning Visualization](https://app.neptune.ai/ds-wook/optiver-prediction/experiments?split=bth&dash=charts&viewId=standard-view)

### Model Architecture
![competition-model](https://user-images.githubusercontent.com/46340424/136811565-958e1ec8-976b-4236-8835-74e2e58b3388.png)

### Benchmark
|Score|CV|Public LB|
|-----|--|------|
|LightGBM(5-group kfold)|0.21558|0.20164|
|LightGBM(10-group kfold)|0.21466|0.20118|
|TabNet(5-kfold)|0.21241|0.20291|
|TabNet(5-group kfold)|0.21386|0.19872|
|FFNN(5-knn+)|0.21253|0.20291|
|FFNN(5-groupkFold)|0.21293|0.20115|

### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
+ xgboost
+ optuna
+ neptune
+ hydra
+ pytorch-tabnet
+ pytorch
+ tensorflow-2.0

### Score
public 120 🥈  
private 1st 121 🥈  
private 2nd 230 🥉  
private 3th 277 🥉  
private 4th 607  
private 5th 290 🥉  
private 6th 334 🥉  
**Final private 🥈**
