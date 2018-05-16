---
## 5月16日更新

经多位网友的共同实验，原方案部分情况下迭代次数稍微不足，导致最终识别率略有小差异，为了相对容易获得论文的最佳结果，对训练方案进行简单更新，实际训练也可根据数据acc训练是否已稳定来判断lr下降的迭代次数：

- 适当增大softmax迭代次数，4万-->12万；
- 增大arcface第一级lr0.1的迭代次数，8万-->12万；

ps：无闲置机器，暂不再更新log。该项目训练步骤，已验证mobilefacnet可复现，良心大作，期待作者后续的研究。

---
## 5月14日更新
更新两个实验测试：

1. arcface_loss_test2-4：接lr0.1已经训练12万次的模型，增强lr0.1步骤的训练8万次，自身acc小幅提升，下降lr后，最终部分结果中lfw最佳结果99.517%，agedb有模型已提升至96.033%+
2. arcface_loss_test2-5：接arcface_loss_test2-4最佳结果的模型进行精调，margin_s=128，延长了lr0.001迭代次数40000，最终部分结果中lfw最佳结果99.500%，agedb有模型已提升至96.150%+，该步骤对lfw未有提升，对agedb提升比较有效，略微超过论文的96.07%；

ps：issues已有人训练出比论文相对更佳的结果，lfw：99.583，agedb：96.083。

---
## 5月11日更新

实验二验证补充实验：增加lr0.1，+40000steps，lr 0.01，+20000steps，初步判断单卡延长迭代步数有效，lfw提升至99.5+的次数增加，agedb可达到95.9+；继续实验延长迭代次数，判断整体最终稳定情况；

---
## 5月10日更新

更新ncnn转换测试步骤；

---
## 5月9日更新
实验二：切换arcface_loss,节选列出lfw最高一组acc结果：

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

2.softmax loss初调：lr0.1，softmax的fc7配置wd_mult=10.0和no_bias=True,训练12万步;

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
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 0 --lr-steps 120000,140000 --wd 0.00004 --fc7-wd-mult 10 --per-batch-size 512 --emb-size 128  --data-dir  ../datasets/faces_ms1m_112x112  --prefix ../models/MobileFaceNet/model-y1-softmax
```
 

3.arcface loss调试：s=64, m=0.5, 起始lr=0.1，在[120000, 160000, 180000, 200000]步处降低lr，总共训练20万步，也可通过判断acc是否稳定后下降lr。该步骤，LFW acc能到0.9955左右，agedb-30 acc能到0.95以上。

切换到src目录下：

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr-steps 120000,160000,180000,200000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 512 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-softmax,60 --prefix ../models/MobileFaceNet/model-y1-arcface
```

4.agedb精调：从3步训练好的模型继续用arcface loss训练，s=128, m=0.5，起始lr=0.001，在[20000, 30000, 40000]步降低lr，这时能得到lfw acc 0.9955左右，agedb-30 acc 0.96左右的最终模型。

```
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --network y1 --ckpt 2 --loss-type 4 --lr 0.001 --lr-steps 20000,30000,40000 --wd 0.00004 --fc7-wd-mult 10 --emb-size 128 --per-batch-size 512 --margin-s 128 --data-dir ../datasets/faces_ms1m_112x112 --pretrained ../models/MobileFaceNet/model-y1-arcface,100 --prefix ../models/MobileFaceNet/model-y1-arcface
```



## ncnn转换步骤

1.去除模型fc7层，切换insightface/deploy目录下

```
python models_slim.py --model ../models/MobileFaceNet/model-y1-arcface,40
```

2.编译最新版本[ncnn](https://github.com/Tencent/ncnn)的mxnet2ncnn工具；
或直接运行mxnet文件夹的mxnet2ncnn.bat脚本


```
mxnet2ncnn.exe model-y1-arcface-symbol.json model-y1-arcface-0000.params mobilefacenet.param mobilefacenet.bin
```
3.速度测试，增加ncnn的benchncnn工程
复制ncnn目录文件到sdcard卡下，运行下列指令
```
adb shell
cp /sdcard/ncnn/* /data/local/tmp/
cd /data/local/tmp/
chmod 0775 benchncnn
./benchncnn 8 8 0
```
ps:该转换与论文相比，缺少BN层合并至Conv层操作，速度和内存占用非最优值，相关测试大致可提速10%。

附高通625粗略测试结果：
四线程：
```
loop_count = 8
num_threads = 4
powersave = 0
   mobilefacenet  min =   41.44  max =  125.16  avg =   61.43
 light_cnn_small  min =   28.45  max =   32.23  avg =   30.10
  LightenedCNN_A  min =  476.45  max =  489.83  avg =  482.24
  LightenedCNN_B  min =  100.70  max =  104.21  avg =  102.52
      squeezenet  min =   64.73  max =   83.19  avg =   68.53
       mobilenet  min =  120.67  max =  128.20  avg =  124.52
    mobilenet_v2  min =  110.60  max =  220.12  avg =  125.52
      shufflenet  min =   42.43  max =   50.24  avg =   44.86
       googlenet  min =  212.73  max =  228.50  avg =  217.07
        resnet18  min =  230.79  max =  285.95  avg =  246.40
         alexnet  min =  402.55  max =  429.71  avg =  414.41
           vgg16  min = 1622.61  max = 1942.04  avg = 1766.67
  squeezenet-ssd  min =  161.68  max =  290.63  avg =  186.38
   mobilenet-ssd  min =  213.72  max =  245.10  avg =  223.55

```
八线程：
```
M6Note:/data/local/tmp $ ./benchncnn 8 8 0
loop_count = 8
num_threads = 8
powersave = 0
   mobilefacenet  min =   27.77  max =   31.11  avg =   28.87
 light_cnn_small  min =   19.77  max =   25.76  avg =   21.89
  LightenedCNN_A  min =  236.45  max =  341.60  avg =  262.61
  LightenedCNN_B  min =   75.45  max =   79.63  avg =   77.04
      squeezenet  min =   44.78  max =   74.40  avg =   49.59
       mobilenet  min =   75.61  max =   93.74  avg =   82.04
    mobilenet_v2  min =   76.06  max =  104.26  avg =   80.32
      shufflenet  min =   30.33  max =   79.53  avg =   36.89
       googlenet  min =  135.60  max =  276.84  avg =  179.23
        resnet18  min =  164.25  max =  224.34  avg =  181.24
         alexnet  min =  225.19  max =  342.46  avg =  250.83
           vgg16  min = 1631.73  max = 2040.82  avg = 1762.53
  squeezenet-ssd  min =  148.15  max =  260.45  avg =  169.15
   mobilenet-ssd  min =  163.48  max =  198.45  avg =  181.06

```

## 相关参考：

[mobilefacenet论文](https://arxiv.org/abs/1804.07573)

[insightface](https://github.com/deepinsight/insightface)

## TODO

- ncnn框架移植mobilefacenet