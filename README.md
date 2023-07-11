# OpenViseme

A simple neural network to compute mouth shape from audio. Data and idea taken from [Magicboomliu/Viseme-Classification](https://github.com/Magicboomliu/Viseme-Classification/tree/main/Data%20Training/DataSet). 

This project is a complete rewrite and imrpovment over Magicboomliu/Viseme-Classification. With better data resampling, cleanup and code quality.

## How to use

1. Download and put the `DataSet` folder from [Viseme-Classification/DataSet](https://github.com/Magicboomliu/Viseme-Classification/tree/main/Data%20Training/DataSet) into the root folder
2. Run `prepare_data.py`. This will generate a `DataSet/mel.npy` and `DataSet/label.npy`. These are the dataset used for training
3. Run `baseline.py` to get a baseline accuracy for low-effot models (using sklearn, to evaulate initial performance for different ML algorithms)
3. Run `train.py` to generate `model.pth` which is trained by ptroch

For me, the baseline (neural network) yields 63% accuracy on testing set
```
Train Accuracy: 0.6464355131983196
Test Accuracy: 0.624623871614844
```

While the PyTorch one yields a stable 63% accuracy (the difference is statistically significant)
```
Epoch: 600, Loss: 1.7374, Test Accuracy: 0.6387, Test Loss: 2.1781
Lowest Loss: 2.1706
Best test Accuracy: 0.6387
```
