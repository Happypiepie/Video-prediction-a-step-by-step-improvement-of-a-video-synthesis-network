# Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network
----------------------------------------------------------------------------------------------
Flow-based Kernel Prior with Application to Blind Super-Resolution (FKP), CVPR2021
This repository is the official PyTorch implementation of Flow-based Kernel Prior with Application to Blind Super-Resolution (arxiv, supp).

🚀 🚀 🚀 News:

Aug. 18, 2021: See our recent work for blind SR: Mutual Affine Network for Spatially Variant Kernel Estimation in Blind Image Super-Resolution (MANet), ICCV2021
Aug. 18, 2021: See our recent work for flow-based generative modelling of image SR: Hierarchical Conditional Flow: A Unified Framework for Image Super-Resolution and Image Rescaling (HCFlow), ICCV2021
Aug. 18, 2021: See our recent work for real-world image SR: Designing a Practical Degradation Model for Deep Blind Image Super-Resolution (BSRGAN), ICCV2021
Kernel estimation is generally one of the key problems for blind image super-resolution (SR). Recently, Double-DIP proposes to model the kernel via a network architecture prior, while KernelGAN employs the deep linear network and several regularization losses to constrain the kernel space. However, they fail to fully exploit the general SR kernel assumption that anisotropic Gaussian kernels are sufficient for image SR. To address this issue, this paper proposes a normalizing flow-based kernel prior (FKP) for kernel modeling. By learning an invertible mapping between the anisotropic Gaussian kernel distribution and a tractable latent distribution, FKP can be easily used to replace the kernel modeling modules of Double-DIP and KernelGAN. Specifically, FKP optimizes the kernel in the latent space rather than the network parameter space, which allows it to generate reasonable kernel initialization, traverse the learned kernel manifold and improve the optimization stability. Extensive experiments on synthetic and real-world images demonstrate that the proposed FKP can significantly improve the kernel estimation accuracy with less parameters, runtime and memory usage, leading to state-of-the-art blind SR results.



Requirements
Python 3.6, PyTorch >= 1.6
Requirements: opencv-python, tqdm
Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5
Quick Run
To run the code without preparing data, run this command:

cd DIPFKP
python main.py --SR --sf 4 --dataset Test
Data Preparation
To prepare testing data, please organize images as data/datasets/DIV2K/HR/0801.png, and run this command:

cd data
python prepare_dataset.py --model DIPFKP --sf 2 --dataset Set5
python prepare_dataset.py --model KernelGANFKP --sf 2 --dataset DIV2K
Commonly used datasets can be downloaded here. Note that KernelGAN/KernelGAN-FKP use analytic X4 kernel based on X2, and do not support X3.

FKP
To train FKP, run this command:

cd FKP
python main.py --train --sf 2
Pretrained FKP and USRNet models are already provided in data/pretrained_models.

DIP-FKP
To test DIP-FKP (no training phase), run this command:

cd DIPFKP
python main.py --SR --sf 2 --dataset Set5
KernelGAN-FKP
To test KernelGAN-FKP (no training phase), run this command:

cd KernelGANFKP
python main.py --SR --sf 2 --dataset DIV2K
Results
Please refer to the paper and supplementary for results. Since both DIP-FKP and KernelGAn-FKP are randomly intialized, different runs may get slightly different results. The reported results are averages of 5 runs.

Citation
@article{liang21fkp,
  title={Flow-based Kernel Prior with Application to Blind Super-Resolution},
  author={Liang, Jingyun and Zhang, Kai and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2103.15977},
  year={2021}
}
License & Acknowledgement
This project is released under the Apache 2.0 license. The codes are based on normalizing_flows, DIP, KernelGAN and USRNet. Please also follow their licenses. Thanks for their great works.
