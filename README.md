# MOAG

The **reproducing** code for ACM MM 2021 paper titled "Multiple Objects-Aware Visual Question Generation."  [[paper]](https://dl.acm.org/doi/abs/10.1145/3474085.3476969)



## Overview

![](./pic/model.png)



## Installation

* PyTorch = 1.12



## Run MOAG

```shell
python train.py
```



## Reference

```shell
@inproceedings{moag,
author = {Xie, Jiayuan and Cai, Yi and Huang, Qingbao and Wang, Tao},
title = {Multiple Objects-Aware Visual Question Generation},
year = {2021},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {4546–4554},
}
```



## News-CcQG

Our Content-controlled Question Generation (**CcQG**)  model is extension model of **MOAG**, details can be found in Neural Network 2023 paper “Visual Question Generation for Explicit Questioning Purposes Based on Target Objects” [[pdf](https://www.sciencedirect.com/science/article/pii/S0893608023004264)]

![image-20231108111739693](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231108111739693.png)

![image-20231108111753821](https://raw.githubusercontent.com/Gary-code/pic/main/img/image-20231108111753821.png)

```
@article{ccqg,
  title={Visual question generation for explicit questioning purposes based on target objects},
  author={Xie, Jiayuan and Chen, Jiali and Fang, Wenhao and Cai, Yi and Li, Qing},
  journal={Neural Networks},
  volume={167},
  pages={638--647},
  year={2023},
  publisher={Elsevier}
}
```
