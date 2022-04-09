# Video-prediction-a-step-by-step-improvement-of-a-video-synthesis-network

[![Watch the video](https://i0.hdslb.com/bfs/archive/536b96ae72420bd5ce49d1d322d6ee62a5c118f2.jpg@640w_400h_1c_!web-space-index-myvideo.webp)](https://www.bilibili.com/video/BV1b5411n73g?spm_id_from=333.999.0.0)

[![Watch the video](https://i0.hdslb.com/bfs/archive/abe6e4deb8fc0b755a81747063de70ee2fa96cde.jpg@640w_400h_1c_!web-space-index-myvideo.webp)](https://www.bilibili.com/video/BV1op4y1s7RD?spm_id_from=333.999.0.0)

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

## Requirements

 * Python 3.6, PyTorch >= 1.6
 
 * Requirements: opencv-python, tqdm
 
 * Platforms: Ubuntu 16.04, cuda-10.0 & cuDNN v-7.5

### Prerequisites
 
What things you need to install the software and how to install them
 
```
Give examples
```
 
### Quick Run
 
A step by step series of examples that tell you how to get a development env running
 
Say what the step will be
 
```
Give the example
```
 
And repeat
 
```
until finished
```
 
End with an example of getting some data out of the system or using it for a little demo
 
## Running the tests
 
Explain how to run the automated tests for this system
 
### Break down into end to end tests
 
Explain what these tests test and why
 
```
Give an example
```
 
### And coding style tests
 
Explain what these tests test and why
 
```
Give an example
```
 
## Deployment
 
Add additional notes about how to deploy this on a live system
 
## Built With
 
* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds
 
## Contributing
 
Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.
 
## Versioning
 
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 
 
## Authors
 
* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
 
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
 
## License
 
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
 
## Acknowledgments
 
* Hat tip to anyone whose code was used
* Inspiration
* etc
