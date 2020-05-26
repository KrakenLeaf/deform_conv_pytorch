## PyTorch Implementation of 3D Deformable Convolution  
This repository implements a 3D version of the defromable convolution architecture proposed in this paper:  
[*Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu and Yichen Wei. Deformable Convolutional Networks. arXiv preprint arXiv:1703.06211, 2017.*](https://arxiv.org/abs/1703.06211) 

This repository is forked from the repository found on: https://github.com/ChunhuanLin/deform_conv_pytorch, by ChunhuanLin, which implements the 2D deformable convolution module. Here, we extend this 
module to handle 3D inputs and allow the deformable convolution to learn 3D offsets. 

### Usage
* The defromable 3D convolution module (as well as the original 2D), i.e., *DeformConv3D*, is defined in `deform_conv_3d.py`.  
* A simple demo is shown in `demo_3d.py`.
* The ipynb file tests against the MXNET 2D implementation only (was not modified at all)

### TODO
 - [x] Memory effeicent implementation - using grid_sample.
 - [x] Visualize offsets

### Notes
* Keep in mind that this demo is simplistic and does not necessariliy allows the module to fully exploit its 3D cabapilities. In this demo, we extend the MNIST dataset such that each digit is now not a number, but a volume. 
  Within each volume, the same digit is either copied or copied and rotated. Classification is now performed based on these 3D inputs. 

### Tested on
* [PyTorch-v1.4.0](http://pytorch.org/docs/1.4.0/)

### License 
* MIT License

Copyright (c) 2020 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
