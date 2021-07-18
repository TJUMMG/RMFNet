#  Residual-Guided Multiscale Fusion Network for Bit-Depth Enhancement
Copyright(c) 2021 Jing Liu
```
If you use this code, please cite the following publication:
J. Liu, X. Wen, W. Nie, Y. Su, P. Jing，and X. Yang, "Residual-Guided Multiscale Fusion Network for Bit-Depth Enhancement", to appear in IEEE Transactions on Circuits and Systems for Video Technology.

```
## Contents

1. [Environment](#1)
2. [Test](#2)


<h3 id="1">Environment</h3>
Our model is tested through the following environment on Ubuntu:

- Python: 3.6.10
- PyTorch: 1.3.1
- opencv：3.4.2

### Testing
We provide four folders "./RMF_4bit/RMF_test_4_16", "./RMF_4bit/RMF_test_4_8", "./RMF_6bit/RMF_test_6_16" and "./RMF_8bit/RMF_test_8_16" to realize 4-bit to 16-bit, 4-bit to 8-bit, 6-bit to 16-bit and 8-bit to 16-bit BDE tasks respectively. When testing, prepare the testing dataset, and modify the dataset path and other related content in the code. We provide an image of UST-HK dataset (16-bit dataset)  and Kodak dataset (8-bit dataset) respectively for sample testing. You can directly test on the sample image by running-

```
$ python main.py \
--test_only
```
If you want to save the predicted high bit-depth images (--save_results) and high bit-depth ground truths (--save_gt), you can  run-

```
$ python main.py \
--test_only \
--save_results \
--save_gt
```

Note: 

1. We provide recovery results of  sample images in the folder "result" of each models. When testing, the predicted results are saved in the folder "test" .
2. The files "./metrics/csnr_bits.m" and "./metrics/cal_ssim_bits.m" are used to calculate PSNR and SSIM, respectively.
3. The package "./comparison_with_SOTA"  is the subjective results of RMFNet and competing algorithms.
