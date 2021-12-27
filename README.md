English | [中文](README-CN.md)

# Introdcution
This is repository with examples for using MegEngine cpp API for model training. In general, we use Python API for model training on PC or server and use Cpp for inference on various devices, including PC, mobile, IOT and so on. However, in some conditions, we need to directly train the model on mobile or IOT, which requires a set of interface to train model with cpp on these devices. [MegEngine](https://github.com/MegEngine/MegEngine) is an effecitive and opensource framework for training and inference of deep-learning. Futhermore, it now provides the support for training with cpp on Linux, Windows, Android, Linux_arm and IOS. Though it is still in the exploratory phase and far from mature, it makes it possible for us to directly train on mobile and IOT. And this repository will show you how to use the APIs of MegEngine to build the process of it.

There is only an example for Mnist dataset with LeNet so far and we may add some other examples later.

# Usages

### Firstly, clone this repository and prepare the dependencies.

```
git clone https://github.com/AsakusaRinne/megengine_cpp_training_example.git
cd megengine_cpp_training_example
./third_party/prepare.sh
./third_party/MegEngine/third_party/prepare.sh
./third_party/MegEngine/third_party/install-mkl.sh
```

### Secondly, configure the environment according to your device and target

Please refer to [Readme for cmake-build of MegEngine](https://github.com/MegEngine/MegEngine/blob/master/scripts/cmake-build/BUILD_README.md)

### Thirdly, compile and build the target. 

There are four kind of targets you can choose, which are host, android_arm, linux_arm and IOS.

The corresponding scripts are ```scripts/host_build.sh```, ```scripts/cross_build_android_arm_train.sh```, ```scripts/cross_build_linux_arm_train.sh```, ```scripts/cross_build_ios_arm_train.sh``` respectively.

Please choose one of these scripts and run it. For example, if we want to build for android_arm, we can run the commands below.

```
./scripts/cross_build_android_arm_train.sh -h # show the usages of the script.
./scripts/cross_build_android_arm_train.sh -d -r # build with debug mode and remove the old directory before building
```

### Finally, copy the executable file to the device and run it.

The executable file is on default in the directory ```install/bin``` of the ```build_dir``` with your selected platform as prefix.

For example, executable file of building with ```android_arm``` and release mode is in ```build_dir/android/arm-xx/Release/install/bin```.

```
./CppTrainingExamples mnist
```

It will ask you input some info of the training, like dype, epochs and the path of dataset. For mnist dataset, one epoch is enough for training.

The following parameters are provided for reference:

```
batchsize: 16
dtype: int8
epochs: 1
```


To get the mnist dataset, you can run the command below.

```
python3 ./mnist/download.py
```

