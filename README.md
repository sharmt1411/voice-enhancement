### 无声带发声转实声AI模型 

尝试训练一个小模型，将气流声转换为标准嗓音。用于帮助无声带人士发声。

因后天因素手术切除声带，一般仍可说话，但是嗓音类似悄悄话一样，只有轻微气流声发声。

### demo
目前初步训练一个1M小模型，可以将气流声转换嗓音

输入音频16k采样率，16位深度
转换为mel图
模型使用conv+conv+conv+lstm+upconv+upconv+upconv实现，数据量大概采集10分钟，预估增大数据集后，参数量仍可减少
一次处理 序列（帧）长度64，在mel图以128hop转换时，对应16K采样率大概0.5s转换一次，仍有优化空间

### 文件说明

src/train 模型训练脚本
src/test-case 测试模型效果的脚本
src/audio_model 模型设计文件
src/dataset 数据集加载文件


### 后续计划

优化延时
尝试部署移动设备

