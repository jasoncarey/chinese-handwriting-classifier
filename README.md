# Chinese Handwritten Character Classification

Dataset taken from [CASIA Online and Offline Chinese Handwriting Databases](https://nlpr.ia.ac.cn/databases/handwriting/Download.html).

A research paper outlining how the samples were produced can be found [here](https://nlpr.ia.ac.cn/databases/download/ICDAR2011-CASIA%20databases.pdf)

We use GB2312 encoding for Chinese characters, where each is represented by a pair of bytes.

## Accuracy Data

| Model Type    | Model Config                              | Epochs | Accuracy |
| ------------- | ----------------------------------------- | ------ | -------- |
| CNN           | 3 convolutional, 2 fully connected layers | 10     | 83.00%   |
| a             | a                                         | a      | 87.11%   |

# TODO: 
- visualisation
- try wandb