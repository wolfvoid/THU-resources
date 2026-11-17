#### Data preparation
Download the data from this link: https://cloud.tsinghua.edu.cn/f/216c7cdca7ca439cbfb3/

#### Model training

Simply run:

```
bash train.sh
```

if you are using linux or macOS. If you are using windows, you can copy the content of the script and run it in your terminal. Remember to replace the dataset_path to your local dataset path.

#### Model evaluation

Run:

```
bash test.sh
```

if you are using linux or macOS. If you are using windows, you can copy the content of the script and run it in your terminal. Remember to replace the dataset_path to your local dataset path and the checkpoint path to your best model checkpoint path.
