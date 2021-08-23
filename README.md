# Optiver-prediction
Apply your data science skills to make financial markets better


### Benchmark
|Score|CV|Public LB|private LB|
|-----|--|------|-------|
|NN(5-kfold)|0.21241|0.20291|-|
|LightGBM(10-kfold)|0.20253|0.20936|-|
|XGBoost(10-kfold)|0.21026|0.21137|-|
|LightGBM(10-group kfold)|0.22188|0.21173|-|
|XGBoost(10-group kfold)|-|-|-|

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
+ omegaconf
