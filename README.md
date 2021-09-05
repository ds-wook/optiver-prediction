# Optiver-prediction
Apply your data science skills to make financial markets better


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|NN(5-kfold)|0.21241|0.20291|-|
|LightGBM(5-kfold)|0.19154||-|
|XGBoost(10-kfold)|0.21026|0.21137|-|
|LightGBM(10-group kfold)|0.22188|0.21173|-|
|XGBoost(10-group kfold)|-|-|-|

### Importance
![split](https://user-images.githubusercontent.com/46340424/131856956-b1164a64-7e97-41cc-a0be-e0f4c214f5f9.png)

### Requirements
+ numpy
+ pandas
+ scikit-learn
+ lightgbm
+ xgboost
+ catboost
+ rgf-python
+ optuna
+ tensorflow==2.1.5
+ neptune
+ hydra
