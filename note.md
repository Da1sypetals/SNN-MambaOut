1. 输入lif的是 t,b,c,h,w，但是输入conv之类的是 n, c, h, w，
    - 输入conv要把t b合并成一个dim；
    - 输入lif的时候再拆开，t=T，T在外面设置（配置文件）
2，过程形状变化：
```
input: x.shape=torch.Size([10, 3, 224, 224])
stem layer: x.shape=torch.Size([10, 3, 224, 224])
gated cnn layer: x.shape=torch.Size([10, 56, 56, 24])
gated cnn layer: x.shape=torch.Size([10, 56, 56, 24])
gated cnn layer: x.shape=torch.Size([10, 56, 56, 24])
downsample layer: x.shape=torch.Size([10, 56, 56, 24])
gated cnn layer: x.shape=torch.Size([10, 28, 28, 48])
gated cnn layer: x.shape=torch.Size([10, 28, 28, 48])
gated cnn layer: x.shape=torch.Size([10, 28, 28, 48])
downsample layer: x.shape=torch.Size([10, 28, 28, 48])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 14, 14, 96])
downsample layer: x.shape=torch.Size([10, 14, 14, 96])
gated cnn layer: x.shape=torch.Size([10, 7, 7, 128])
gated cnn layer: x.shape=torch.Size([10, 7, 7, 128])
gated cnn layer: x.shape=torch.Size([10, 7, 7, 128])
mlp layer: x.shape=torch.Size([10, 128])
torch.Size([10, 1000])
```
3. 创新点：quantize？T=N_quantize