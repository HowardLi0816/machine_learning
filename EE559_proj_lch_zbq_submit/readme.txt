Requirments:
imbalanced-learn: 0.7.0 (conda install -c conda-forge imbalanced-learn)
pytorch-cuda:11.7

Run the best model:
1. cd ./model/MLP_percep
2. For training the best model and test it on test set:
python .\credit_MLP.py --norm=True --data_augment=False --ufs=True --ufs_k=20 --rfe=False --rfe_k=10 --sfs=False --sfs_k=22 --pca=False --pca_n_component=10 --lda=False --non_linear=True --poly=2 --lr=1e-4 --layer_model 18 --max_epoch=1000 --is_test=True
