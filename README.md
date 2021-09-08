# Optiver-prediction
Apply your data science skills to make financial markets better


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|LightGBM(5-kfold)|0.19076|0.20465|-|
|XGBoost(10-kfold)|0.21026|0.21137|-|
|TabNet(5-kfold)|0.21241|0.20291|-|
|LightGBM(5-group kfold)|0.215995|0.20189|-|
|XGBoost(10-group kfold)|0.22607|-|-|

### Importance
![split](https://user-images.githubusercontent.com/46340424/131856956-b1164a64-7e97-41cc-a0be-e0f4c214f5f9.png)

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
