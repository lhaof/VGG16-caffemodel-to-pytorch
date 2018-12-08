from __future__ import division
import sys
caffe_root = '../caffe_dss-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from PIL import Image

class vgg16net(nn.Module):
	def __init__(self):
		super(vgg16net, self).__init__()

		self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv1_1
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv1_2
									nn.ReLU(inplace=True) 
									)
		self.conv2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool1
									nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv2_1
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv2_2
									nn.ReLU(inplace=True) 
									)
		self.conv3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool3
									nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv3_1
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv3_2
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv3_3
									nn.ReLU(inplace=True) 
									)
		self.conv4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool3
									nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv4_1 
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv4_2
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv4_3
									nn.ReLU(inplace=True) 
									)
		self.conv5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool4
									nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv5_1 
									nn.ReLU(inplace=True),
									nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv5_2
									nn.ReLU(inplace=True), 
									nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), # conv5_3
									nn.ReLU(inplace=True) 
									)

		self.pool5 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True), # pool5
									)

		self.fc6 = nn.Sequential(nn.Linear(512*7*7, 4096),
								nn.ReLU(inplace=True))

		self.fc7 = nn.Sequential(nn.Linear(4096, 4096),
								nn.ReLU(inplace=True))

		self.fc8 = nn.Sequential(nn.Linear(4096, 1000))  

		self.prob = nn.Softmax(1)

	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3) 
		conv5 = self.conv5(conv4) 
		pool5 = self.pool5(conv5) 
		pool5 = pool5.view(-1, 512*7*7)
		fc6 = self.fc6(pool5)
		fc7 = self.fc7(fc6) 
		fc8 = self.fc8(fc7) 
		prob = self.prob(fc8)
		return prob 

caffe.set_mode_gpu()
caffe.set_device(0)
caffeproto = 'VGG_ILSVRC_16_layers_deploy.prototxt'
caffeweight = 'VGG_ILSVRC_16_layers.caffemodel'
caffenet = caffe.Net(caffeproto, caffeweight, caffe.TEST)
print caffenet.params.keys()

pthnet = vgg16net()
pthnet.cuda()
print pthnet.state_dict().keys()

pthnet.state_dict()['conv1.0.weight'].copy_(torch.from_numpy(caffenet.params['conv1_1'][0].data))
pthnet.state_dict()['conv1.0.bias'].copy_(torch.from_numpy(caffenet.params['conv1_1'][1].data))
pthnet.state_dict()['conv1.2.weight'].copy_(torch.from_numpy(caffenet.params['conv1_2'][0].data))
pthnet.state_dict()['conv1.2.bias'].copy_(torch.from_numpy(caffenet.params['conv1_2'][1].data))

pthnet.state_dict()['conv2.1.weight'].copy_(torch.from_numpy(caffenet.params['conv2_1'][0].data))
pthnet.state_dict()['conv2.1.bias'].copy_(torch.from_numpy(caffenet.params['conv2_1'][1].data))
pthnet.state_dict()['conv2.3.weight'].copy_(torch.from_numpy(caffenet.params['conv2_2'][0].data))
pthnet.state_dict()['conv2.3.bias'].copy_(torch.from_numpy(caffenet.params['conv2_2'][1].data))

pthnet.state_dict()['conv3.1.weight'].copy_(torch.from_numpy(caffenet.params['conv3_1'][0].data))
pthnet.state_dict()['conv3.1.bias'].copy_(torch.from_numpy(caffenet.params['conv3_1'][1].data))
pthnet.state_dict()['conv3.3.weight'].copy_(torch.from_numpy(caffenet.params['conv3_2'][0].data))
pthnet.state_dict()['conv3.3.bias'].copy_(torch.from_numpy(caffenet.params['conv3_2'][1].data))
pthnet.state_dict()['conv3.5.weight'].copy_(torch.from_numpy(caffenet.params['conv3_3'][0].data))
pthnet.state_dict()['conv3.5.bias'].copy_(torch.from_numpy(caffenet.params['conv3_3'][1].data))

pthnet.state_dict()['conv4.1.weight'].copy_(torch.from_numpy(caffenet.params['conv4_1'][0].data))
pthnet.state_dict()['conv4.1.bias'].copy_(torch.from_numpy(caffenet.params['conv4_1'][1].data))
pthnet.state_dict()['conv4.3.weight'].copy_(torch.from_numpy(caffenet.params['conv4_2'][0].data))
pthnet.state_dict()['conv4.3.bias'].copy_(torch.from_numpy(caffenet.params['conv4_2'][1].data))
pthnet.state_dict()['conv4.5.weight'].copy_(torch.from_numpy(caffenet.params['conv4_3'][0].data))
pthnet.state_dict()['conv4.5.bias'].copy_(torch.from_numpy(caffenet.params['conv4_3'][1].data))

pthnet.state_dict()['conv5.1.weight'].copy_(torch.from_numpy(caffenet.params['conv5_1'][0].data))
pthnet.state_dict()['conv5.1.bias'].copy_(torch.from_numpy(caffenet.params['conv5_1'][1].data))
pthnet.state_dict()['conv5.3.weight'].copy_(torch.from_numpy(caffenet.params['conv5_2'][0].data))
pthnet.state_dict()['conv5.3.bias'].copy_(torch.from_numpy(caffenet.params['conv5_2'][1].data))
pthnet.state_dict()['conv5.5.weight'].copy_(torch.from_numpy(caffenet.params['conv5_3'][0].data))
pthnet.state_dict()['conv5.5.bias'].copy_(torch.from_numpy(caffenet.params['conv5_3'][1].data))

pthnet.state_dict()['fc6.0.weight'].copy_(torch.from_numpy(caffenet.params['fc6'][0].data)) 
pthnet.state_dict()['fc6.0.bias'].copy_(torch.from_numpy(caffenet.params['fc6'][1].data)) 
pthnet.state_dict()['fc7.0.weight'].copy_(torch.from_numpy(caffenet.params['fc7'][0].data)) 
pthnet.state_dict()['fc7.0.bias'].copy_(torch.from_numpy(caffenet.params['fc7'][1].data)) 
pthnet.state_dict()['fc8.0.weight'].copy_(torch.from_numpy(caffenet.params['fc8'][0].data)) 
pthnet.state_dict()['fc8.0.bias'].copy_(torch.from_numpy(caffenet.params['fc8'][1].data)) 

imname = './dataset/HKU-IS/imgs/0004.png'
im = Image.open(imname).convert('RGB')
im = im.resize((224,224)) 
im = np.array(im).astype(np.float32) 
im = im[:, :, ::-1]
im -= np.array((104.00699, 116.66877, 122.67892))
im = im.transpose((2,0,1))
im = np.ascontiguousarray(im) 
im = np.expand_dims(im, axis=0) 

caffenet.blobs['data'].data[...] = im 
caffenet.forward()
caffeout = caffenet.blobs['prob'].data.copy()
print caffeout.shape, caffeout.min(), caffeout.mean(), caffeout.max()

pthout = pthnet.forward(torch.from_numpy(im).cuda())
pthout = pthout.cpu().data.numpy()
print pthout.shape, pthout.min(), pthout.mean(), pthout.max()
print 'done'