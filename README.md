# Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network

[![Watch the video](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/lvideo.gif)](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/lvideo.gif)
[![Watch the video](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/ccideo.gif)](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/ccideo.gif)

[![Watch the video](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/rlvideo.gif)](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/rlvideo.gif)
[![Watch the video](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/rvideo.gif)](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/rvideo.gif)
# Abstract
Although focusing on the field of video generation has made some progress in network performance and computationalefficiency, 
there is still much room for improvement in terms of the predicted frame number and clarity. In this paper, a depthlearning 
model is proposed to predict future video frames. The model can predict video streams with complex pixel distributionsof up 
to 32 frames. Our framework is mainly composed of two modules: a fusion image prediction generator and an image-videotranslator. 
The fusion picture prediction generator is realized by a U-Net neural network built by a 3D convolution, and theimage-video 
translator is composed of a conditional generative adversarial network built by a 2D convolution network. In theproposed 
framework, given a set of fusion images and labels, the image picture prediction generator can learn the pixeldistribution 
of the fitted label pictures from the fusion images. The image-video translator then translates the output of the fused
image prediction generator into future video frames. In addition, this paper proposes an accompanying convolution model and
corresponding algorithm for improving image sharpness. Our experimental results prove the effectiveness of this framework.
![This is an image](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/model.png)

# Requirements

 * Python 3.6, PyTorch >= 1.6
 
 * Requirements: opencv-python, tqdm
 
 * Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5

# Datasets
 
### Train、Val Dataset
The train and val datasets are sampled from Dvind-2017. Train dataset has 16700 images and Val dataset has 425 images. Download the datasets from here, and then extract it into data directory. Finally run
 
```
python data_processpy

optional arguments:
--upscale_factor      super resolution upscale factor [default value is 3]
```
 
# Train
 
```
python train.py

optional arguments:
--upscale_factor       upscale factor [default value is 3]
--is_real_time         real time to show [default value is False]
--delay_time           delay time to show [default value is 1]
--model_name           model name [default value is epoch_3_100.pt]
```
 
 
## Running the tests
 
 
```
python test.py

```

The result in results folder.
 
# Benchmarks
Adam optimizer were used with learning rate scheduling between epoch 30 and epoch 80.

Upscale Factor = 2

Epochs with batch size of 64 takes ~1 minute on a NVIDIA GeForce 3090 GPU.

![This is an image](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/PSNR_BIKE.png)
![This is an image](https://github.com/Happypiepie/Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network/blob/main/SSIM_BIKE.png)
In the case of three 16consecutive fused picturesselected at random, pix2pixHD is
quantitatively compared with the model in this paper. Given 16 input frames, the model recursively generates 32 output frames. In this paper, the wake-upstatistics are all from the 16th frame generated. Left: the PSNRvalue is evaluated. Right: the SSIM value is evaluated

# Cite
 
 
```
@article{jing2021video,
  title={Video prediction: a step-by-step improvement of a video synthesis network},
  author={Jing, Beibei and Ding, Hongwei and Yang, Zhijun and Li, Bo and Bao, Liyong},
  journal={Applied Intelligence},
  pages={1--13},
  year={2021},
  publisher={Springer}
}

```










 

