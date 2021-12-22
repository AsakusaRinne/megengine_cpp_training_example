[English](README.md) | 中文

# 介绍
本项目用于展示MegEngine的端上训练功能，目前仅给出了基于Mnist数据集使用LeNet进行端上训练的demo。

[MegEngine](https://github.com/MegEngine/MegEngine)是一个快速、可拓展、易于使用且支持自动求导的深度学习框，不仅如此，现在MegEngine还对端上训练提供了支持，端上训练主要适用于移动端和IOT等场景，在一些情况下不方便将采集到的数据通过网络传回服务端进行训练，比如对人脸数据、指纹等的采集涉及到隐私和法律问题，这时候就需要在移动端或IOT设备上对模型进行训练。

目前MegEngine端上训练还处于初期的探索阶段，可能还不太成熟，期待MegEngine之后会推出更强大而友好的端上训练API！

# 使用

### 首先，克隆本项目并运行脚本准备依赖项

提示：国内可能会克隆一些子项目比较慢，甚至失败，请多尝试几次或者使用代理。

```
git clone https://github.com/AsakusaRinne/megengine_cpp_training_examples.git
./third_party/prepare.sh
./third_party/MegEngine/third_party/prepare.sh
./third_party/MegEngine/third_party/install-mkl.sh
```

### 然后，根据设备以及目标平台进行环境的配置，请根据下面的MegEngine文档进行

[MegEngine cmake-build配置文档](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md)

### 编译并运行 

这里有四种脚本可以进行自动编译与生成，分别是host, android_arm, linux_arm和IOS，请根据自己需要的目标平台选取脚本并运行。如果是其它的平台，也可以自己手动进行配置并编译与生成。

脚本分别为 ```scripts/host_build.sh```, ```scripts/cross_build_android_arm_train.sh```, ```scripts/cross_build_linux_arm_train.sh```和```scripts/cross_build_ios_arm_train.sh```。

比如如果我们要在Arm架构的Android平台上运行demo，我们可以运行以下命令，其中```-h```用来查询脚本的可选命令。

```
./scripts/cross_build_android_arm_train.sh -h # show the usages of the script.
./scripts/cross_build_android_arm_train.sh -d -r # build with debug mode and remove the old directory before building
```

### 最后，将生成好的可执行文件拷贝到目标设备并执行

其中，可执行文件默认存在于```build_dir```中与目标平台对应的目录中的```install/bin```文件夹下，也可以自行修改```CMakeLists.txt```中的```install```命令内容来指定安装目录。

比如目标平台是```android_arm```且使用release模式时，处于```build_dir/android/arm-xx/Release/install/bin```路径下。

```
./CppTrainingExamples mnist
```

程序会要求输入一些和训练有关的信息，包括数据类型、训练轮数、数据集路径等。

以下参数作为参考：

```
batchsize: 16
dtype: int8
epochs: 1
```


可以通过以下命令下载mnist数据集：

```
python3 ./mnist/dataset/download.py
```
