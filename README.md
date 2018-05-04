
## 前言

本文主要记录下复现mobilefacenet的流程，参考mobilefacenet作者月生给的基本流程，基于insightface的4月27日
```
4bc813215a4603474c840c85fa2113f5354c7180
```
版本代码在P40单显卡训练调试。

## 训练步骤
1. 拉取配置[insightface](http://note.youdao.com/)工程的基础环境；
2. 修改src/train_softmax.py文件部分代码；
166-168行：
```
 elif args.network[0]=='y':
    print('init mobilefacenet', args.num_layers)
    embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom = args.bn_mom, wd_mult = args.fc7_wd_mult)
```
修改为：

```
 elif args.network[0]=='y':
    print('init mobilefacenet', args.num_layers)
    embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom = args.bn_mom, no_bias = True，wd_mult = args.fc7_wd_mult)
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

3.softmax loss初调：lr0.1，fc7配置wd_mult=10,训练4万步;

切换到src目录下，运行：
```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --per-batch-size 512 --emb-size 128 --fc7-wd-mult 10  --data-dir  ../data/faces_ms1m_112x112  --prefix ../models/MobileFaceNet/model-y1-softmax
```
 

4.softmax模型去除fc7层：

切换到deploy目录下
```
python model_slim.py --model ../models/MobileFaceNet/model-y1-softmax,20
```

5.arcface loss调试：s=64, m=0.5, 起始lr=0.1，在[80000, 120000, 140000, 160000]步处降低lr，总共训练16万步。这时，LFW acc能到0.9955左右，agedb-30 acc能到0.959以上。

切换到src目录下：

```
CUDA_VISIBLE_DEVICES='0' python -u /data/src/train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr-steps 80000,120000,140000,160000 --emb-size 128 --per-batch-size 512 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,0 --prefix ../models/MobileFaceNet/model-y1-arcface
```

6.agedb精调：从5步训练好的模型继续用arcface loss训练，s=128, m=0.5，起始lr=0.001，在[20000, 30000, 40000]步降低lr，这时能得到lfw acc 0.9955左右，agedb-30 acc 0.96左右的最终模型。

```
CUDA_VISIBLE_DEVICES='0' python -u /data/src/train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001 --lr-steps 20000,30000,40000 --emb-size 128 --per-batch-size 512 --margin-s 128 --data-dir ../data/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,80 --prefix ../models/MobileFaceNet/model-y1-arcface
```

## 相关参考：

[mobilefacenet论文](https://arxiv.org/abs/1804.07573)

[insightface](https://github.com/deepinsight/insightface)

## TODO

- 三天后上传该方案验证logo
- 训练模型识别率等测试
- ncnn框架移植mobilefacenet