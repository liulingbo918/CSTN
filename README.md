# Taxi Origin-Destination Demand Prediction


This is a Tensorflow implementation of **[Contextualized Spatial-Temporal Network for Taxi Origin-Destination Demand Prediction](https://ieeexplore.ieee.org/abstract/document/8720246)** published in IEEE Transactions on Intelligent Transportation Systems. In this paper, we present a challenging and worth-exploring task, called taxi origin-destination demand prediction, which aims at predicting the taxi demand between all-region pairs in a future time interval. We address this problem with a novel contextualized spatial-temporal network (CSTN), which consists of three components for the modeling of local spatial context (LSC), temporal evolution context (TEC), and global correlation context (GCC), respectively.

If you use this code for your research, please cite our work:

```
@article{liu2019contextualized,
  title={Contextualized Spatial-Temporal Network for Taxi Origin-Destination Demand Prediction},
  author={Liu, Lingbo and Qiu, Zhilin and Li, Guanbin and Wang, Qing and Ouyang, Wanli and Lin, Liang},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```

## Requirements
```
Keras==2.1.4
pandas==0.17.1
tensorflow==1.14.0
numpy==1.16.2
```

## Dataset Preprocessing
download NYC-TOD.tar.gz with following links and put it into folder  ```NYC-TOD/```.
- [Dropbox](https://www.dropbox.com/s/ft4i0i0bysoox55/NYC-TOD.tar.gz?dl=0/)
- [BaiduYun, password：j7ok](https://pan.baidu.com/s/1BiBYsgbfng83xK-D3wIbCw)

## Training and Testing
```bash
sh train_test.sh
```




