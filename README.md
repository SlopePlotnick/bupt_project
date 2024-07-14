# Evaluation of Classification algorithms for Distributed Denial of Service Attack Detection

本代码是对论文**Evaluation of Classification algorithms for Distributed Denial of Service Attack Detection**中平衡数据集算法评估部分的复现与改进。对比评估了多个算法针对分布式拒绝服务攻击检测的分类性能。

## Folders

```
.
├── README.md
├── feature_selected.csv 选取的用于训练的25个最重要的特征
├── main.py 运行脚本
└── results.csv 模型评估结果
```

## Data

+ 使用CICDDoS2019数据集，获取地址：https://www.unb.ca/cic/datasets/ddos-2019.html

+ 预处理后会生成如下文件：
  + `export_dataframe_proc.csv`存放用于训练的预处理数据
  + `export_tests_proc.csv`存放用于测试的预处理数据

## Models

+ Support Vector Machine
+ Logistic Regression
+ K Nearest Neighbor(k == 3)
+ Random Forest
+ Decision Tree
+ Naive Bayes

## Run

```shell
python3 main.py
```
