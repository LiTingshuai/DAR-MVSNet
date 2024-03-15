# DAR-MVSNet: Dual-attention-guided and Residual Network for Multi-View Stereo

## Contact Author
If you have any questions, please contact the author `litingshuai666@gmail.com`

## Change Log
* Mar 14, 2024: Initialize repo
* Mar 15, 2024: Modify code
## Installation
Our code is tested with Python==3.7.12, PyTorch==1.91+cu111, CUDA==11.2 on Ubuntu-18.04 with NVIDIA A100.

To use DAR-MVSNet, clone this repo:

```
git clone https://github.com/LiTingshuai/DAR-MVSNet.git
cd DAR-MVSNet
```
Creating an environment using anaconda:
```
conda create -n darmvsnet python=3.7
conda activate darmvsnet
pip install -r requirements.txt
```
## Data preparation
In DAR-MVSNet, we mainly use [DTU](https://roboimagedata.compute.dtu.dk/) for training and validation, [BlendedMVS](https://github.com/YoYo000/BlendedMVS/) for finetuning, and [Tanks and Temples](https://www.tanksandtemples.org/) for final testing. You can prepare the corresponding data by following the instructions below.

### DTU
For DTU training set, you can download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and unzip them to construct a dataset folder like:
```
dtu_training
 ├── Cameras
 ├── Depths
 ├── Depths_raw
 └── Rectified
```
For DTU testing set, you can download the preprocessed [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the test data folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.

###   BlendedMVS
We use the [low-res set](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) of BlendedMVS dataset for both training and testing. You can download the [low-res set](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) from [orignal BlendedMVS](https://github.com/YoYo000/BlendedMVS) and unzip it to form the dataset folder like below:

```
BlendedMVS
 ├── 5a0271884e62597cdee0d0eb
 │     ├── blended_images
 │     ├── cams
 │     └── rendered_depth_maps
 ├── 59338e76772c3e6384afbb15
 ├── 59f363a8b45be22330016cad
 ├── ...
 ├── all_list.txt
 ├── training_list.txt
 └── validation_list.txt
```

###   Tanks and Temples
Download our preprocessed [Tanks and Temples dataset](https://drive.google.com/file/d/1IHG5GCJK1pDVhDtTHFS3sY-ePaK75Qzg/view?usp=sharing) and unzip it to form the dataset folder like below:
```
tankandtemples
 ├── advanced
 │  ├── Auditorium
 │  ├── Ballroom
 │  ├── ...
 │  └── Temple
 └── intermediate
        ├── Family
        ├── Francis
        ├── ...
        └── Train
```
## Training 
### Training on DTU
Set the configuration in `scripts/train.sh`:

* Set `MVS_TRAINING` as the path of DTU training set.
* Set `LOG_DIR` to save the checkpoints.
* Change `NGPUS` to suit your device.
* We use `torch.distributed.launch` by default.
To train your own model, just run:
```
bash scripts/train.sh
```
You can conveniently modify more hyper-parameters in `scripts/train.sh` according to the argparser in `train.py`, such as `summary_freq`, `save_freq`, and so on.

###Finetune on BlendedMVS
For a fair comparison with other SOTA methods on Tanks and Temples benchmark, we finetune our model on BlendedMVS dataset after training on DTU dataset.

Set the configuration in `scripts/train_bld_fintune.sh`:

* Set `MVS_TRAINING` as the path of BlendedMVS dataset.
* Set `LOG_DIR` to save the checkpoints and training log.
* Set `CKPT` as path of the loaded `.ckpt` which is trained on DTU dataset.
To finetune your own model, just run:
```
bash scripts/train_bld_fintune.sh
```
## Testing
You can use your own model to test according to the following procedure or use our [pre-trained model](https://drive.google.com/drive/folders/12dTNW3FuNclKXP7Xlq5nvqneDtdewp9m?usp=sharing) for testing.

###  Testing on DTU

**Important Tips:** to reproduce our reported results, you need to:
* compile and install the modified `gipuma` from [Yao Yao](https://github.com/YoYo000/fusibile) as introduced below
* make sure you install the right version of python and pytorch, use some old versions would throw warnings of the default action of `align_corner` in several functions, which would affect the final results
* be aware that we only test the code on NVIDIA A100 and Ubuntu 18.04, other devices and systems might get slightly different results
* make sure that you use the `model_dtu.ckpt` for testing


To start testing, set the configuration in ``scripts/test_dtu.sh``:
* Set ``TESTPATH`` as the path of DTU testing set.
* Set ``TESTLIST`` as the path of test list (.txt file).
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save results.

Run:
```
bash scripts/test_dtu.sh
```
**Note:** You can use the `gipuma` fusion method or `normal` fusion method to fuse the point clouds. **In our experiments, we use the `gipuma` fusion method by default**. `Normal` fusion may lead to reduced results.

<!-- The simple instruction for installing and compiling `gipuma` can be found [here](https://github.com/YoYo000/MVSNet#post-processing).  The installed gipuma is a modified version from [Yao Yao](https://github.com/YoYo000/fusibile).-->
To install the `gipuma`, clone the modified version from [Yao Yao](https://github.com/YoYo000/fusibile).
Modify the line-10 in `CMakeLists.txt` to suit your GPUs. Othervise you would meet warnings when compile it, which would lead to failure and get 0 points in fused point cloud. For example, if you use A100 GPU, modify the line-10 to:
```
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_80,code=sm_80)
```
If you use other kind of GPUs, please modify the arch code to suit your device (`arch=compute_XX,code=sm_XX`).
Then install it by `cmake .` and `make`, which will generate the executable file at `FUSIBILE_EXE_PATH`.
Please note 



For quantitative evaluation on DTU dataset, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36). Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```
In ``DTU-MATLAB/BaseEvalMain_web.m``, set `dataPath` as path to `SampleSet/MVS Data/`, `plyPath` as directory that stores the reconstructed point clouds and `resultsPath` as directory to store the evaluation results. Then run ``DTU-MATLAB/BaseEvalMain_web.m`` in matlab.

We also upload our final point cloud results to [here](https://drive.google.com/drive/folders/1Pcc3OF_swEhgdhkjeq7YUtUbJAiwWJdJ?usp=sharing). You can easily download them and evaluate them using the `MATLAB` scripts, the results look like:


| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.351     | 0.255      | 0.303        |

In addition, the files we obtained during the Matlab evaluation can be downloaded [here](https://drive.google.com/drive/folders/1r3rMIZNbTLrdWI-bAgJkCO570GS79cQj?usp=sharing).


###  Testing on Tanks and Temples
We recommend using the finetuned models (``model_bld.ckpt``) to test on Tanks and Temples benchmark.

Similarly, set the configuration in ``scripts/test_tnt.sh``:
* Set ``TESTPATH`` as the path of intermediate set or advanced set.
* Set ``TESTLIST`` as the path of test list (.txt file).
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save resutls.

To generate point cloud results, just run:
```
bash scripts/test_tnt.sh
```

For quantitative evaluation, you can upload your point clouds to [Tanks and Temples benchmark](https://www.tanksandtemples.org/).

##Acknowledgments
We borrow some code from [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet),  [AA-RMVSNet](https://github.com/QT-Zhu/AA-RMVSNet) and [TransMVSNet](https://github.com/megvii-research/TransMVSNet?tab=readme-ov-file). We thank the authors for releasing the source code.