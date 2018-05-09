
---
## 5月9日更新
实验二：切换arcface_loss训练,节选列出lfw最高一组acc结果：

```
[2018-05-09 02:28:45]  lr-batch-epoch: 0.01 534 15
[2018-05-09 02:28:45]  testing verification..
[2018-05-09 02:28:58]  (12000, 128)
[2018-05-09 02:28:58]  infer time 12.946839
[2018-05-09 02:29:02]  [lfw][112000]XNorm: 11.147283
[2018-05-09 02:29:02]  [lfw][112000]Accuracy-Flip: 0.99517+-0.00450
[2018-05-09 02:29:02]  testing verification..
[2018-05-09 02:29:18]  (14000, 128)
[2018-05-09 02:29:18]  infer time 15.957752
[2018-05-09 02:29:23]  [cfp_fp][112000]XNorm: 9.074075
[2018-05-09 02:29:23]  [cfp_fp][112000]Accuracy-Flip: 0.88457+-0.01533
[2018-05-09 02:29:23]  testing verification..
[2018-05-09 02:29:35]  (12000, 128)
[2018-05-09 02:29:35]  infer time 12.255588
[2018-05-09 02:29:39]  [agedb_30][112000]XNorm: 11.038146
[2018-05-09 02:29:39]  [agedb_30][112000]Accuracy-Flip: 0.95067+-0.00907
```

目前离论文要求识别率已非常接近，下组实验增加迭代轮数，判断是否因为单卡原因；

---
## 5月7日更新

实验一，目前测试效果不佳，softmax预训练未达到预期在lfw上98+的识别率，待排查及进一步实验。如何在lr0.1下达到一个合理的预训练区间，对后续是否能训练到最优识别率影响较大。

实验二：

论文指出：
```
We set the weight decay parameter to be 4e-5, except the weight decay 
parameter of the last layers after the global operator (GDConv or GAPool) being 4e-4. 
```

修复错误：--wd设置0.00004，--fc7-wd-mult设置10，重新进行试验；
<<<<<<< HEAD

实验日志：softmax训练的acc持续提升，lfw上99+，转下一步训练；
=======
>>>>>>> 1b5e6cdb8ae9009c8788c941d310331a87e8cc3e

实验日志：softmax训练的acc持续提升，lfw上99+，转下一步训练；

---

## 前言

本文主要记录下复现mobilefacenet的流程，参考mobilefacenet作者月生给的基本流程，基于insightface的4月27日
```
4bc813215a4603474c840c85fa2113f5354c7180
```
版本代码在P40单显卡训练调试。

## 训练步骤
1.拉取配置[insightface](https://github.com/deepinsight/insightface)工程的基础环境；

2.softmax loss初调：lr0.1，softmax的fc7配置wd_mult=10.0和no_bias=True,训练4万步;

切换到src目录下，修改train_softmax.py：
179-182行：
```
  if args.loss_type==0: #softmax
    _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
```
修改为：

```
  if args.loss_type==0: #softmax
    #_bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
    # fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, bias = _bias, num_hidden=args.num_classes, name='fc7')
    fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias=True, num_hidden=args.num_classes, name='fc7')
```

363行：

```
 if args.network[0]=='r' or args.network[0]=='y':
```
修改为：

```
 if args.network[0]=='r' :
```
这样保证uniform初始化；


运行：
```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --lr-steps 80000,120000 --wd 0.00004 --fc7-wd-mult 10 --per-batch-size 512 --emb-size 128  --data-dir  ../datasets/faces_ms1m_112x112  --prefix ../models/MobileFaceNet/model-y1-softmax
```
 

3.arcface loss调试：s=64, m=0.5, 起始lr=0.1，在[80000, 120000, 140000, 160000]步处降低lr，总共训练16万步。这时，LFW acc能到0.9955左右，agedb-30 acc能到0.959以上。

切换到src目录下：

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr-steps 80000,120000,140000,160000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 512 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,20 --prefix ../models/MobileFaceNet/model-y1-arcface
```

4.agedb精调：从3步训练好的模型继续用arcface loss训练，s=128, m=0.5，起始lr=0.001，在[20000, 30000, 40000]步降低lr，这时能得到lfw acc 0.9955左右，agedb-30 acc 0.96左右的最终模型。

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001 --lr-steps 20000,30000,40000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 512 --margin-s 128 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,80 --prefix ../models/MobileFaceNet/model-y1-arcface
```

## 相关参考：

[mobilefacenet论文](https://arxiv.org/abs/1804.07573)

[insightface](https://github.com/deepinsight/insightface)

## TODO

- ncnn框架移植mobilefacenet