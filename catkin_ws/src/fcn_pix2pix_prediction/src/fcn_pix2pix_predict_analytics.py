#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
import os
import rospkg
import math
import time
import sys
import PIL
import pandas as pd
import scipy.misc
import random
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import message_filters
from datetime import datetime

from torchvision import transforms, utils, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.vgg import VGG
from sklearn.metrics import confusion_matrix
from subt_msgs.msg import *


class FCN16s(nn.Module):

	def __init__(self, pretrained_net, n_class):
		super(FCN16s, self).__init__()
		self.n_class = n_class
		self.pretrained_net = pretrained_net
		self.relu = nn.ReLU(inplace=True)
		self.deconv1 = nn.ConvTranspose2d(
			512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn1 = nn.BatchNorm2d(512)
		self.deconv2 = nn.ConvTranspose2d(
			512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.deconv3 = nn.ConvTranspose2d(
			256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn3 = nn.BatchNorm2d(128)
		self.deconv4 = nn.ConvTranspose2d(
			128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.deconv5 = nn.ConvTranspose2d(
			64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
		self.bn5 = nn.BatchNorm2d(32)
		self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

	def forward(self, x):
		output = self.pretrained_net(x)
		x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
		x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

		# size=(N, 512, x.H/16, x.W/16)
		score = self.relu(self.deconv1(x5))
		# element-wise add, size=(N, 512, x.H/16, x.W/16)
		score = self.bn1(score + x4)
		# size=(N, 256, x.H/8, x.W/8)
		score = self.bn2(self.relu(self.deconv2(score)))
		# size=(N, 128, x.H/4, x.W/4)
		score = self.bn3(self.relu(self.deconv3(score)))
		# size=(N, 64, x.H/2, x.W/2)
		score = self.bn4(self.relu(self.deconv4(score)))
		score = self.bn5(self.relu(self.deconv5(score))
						 )  # size=(N, 32, x.H, x.W)
		# size=(N, n_class, x.H/1, x.W/1)
		score = self.classifier(score)

		return score


class VGGNet(VGG):
	def __init__(self, cfg, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
		super(VGGNet, self).__init__(self.make_layers(cfg[model]))
		ranges = {
			'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
			'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
			'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
			'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
		}
		self.ranges = ranges[model]

		if pretrained:
			exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

		if not requires_grad:
			for param in super().parameters():
				param.requires_grad = False

		if remove_fc:  # delete redundant fully-connected layer params, can save memory
			del self.classifier

		if show_params:
			for name, param in self.named_parameters():
				print(name, param.size())

	def forward(self, x):
		output = {}

		# get the output of each maxpooling layer (5 maxpool in VGG net)
		for idx in range(len(self.ranges)):
			for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
				x = self.features[layer](x)
			output["x%d" % (idx+1)] = x
		return output

	def make_layers(self, cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d,
							   nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)


class FCN_Pix2Pix_PREDICT():
	def __init__(self):

		self.bridge = CvBridge()
		r = rospkg.RosPack()
		self.path = r.get_path('fcn_pix2pix_prediction')
		self.cfg = {
			'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
			'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
			'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
		}
		self.means = np.array([103.939, 116.779, 123.68]) / \
			255.  # mean of three channels in the order of BGR
		self.h, self.w = 480, 640
		# self.resize_count = 0
		self.n_class = 5
		model_name = "fcn_pix2pix.pkl"
		self.vgg_model = VGGNet(self.cfg, requires_grad=True, remove_fc=True)
		self.fcn_model = FCN16s(
			pretrained_net=self.vgg_model, n_class=self.n_class)

		use_gpu = torch.cuda.is_available()
		num_gpu = list(range(torch.cuda.device_count()))
		rospy.loginfo("Cuda available: %s", use_gpu)

		if use_gpu:
			ts = time.time()
			self.vgg_model = self.vgg_model.cuda()
			self.fcn_model = self.fcn_model.cuda()
			self.fcn_model = nn.DataParallel(
				self.fcn_model, device_ids=num_gpu)
			print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
		state_dict = torch.load(os.path.join(
			self.path, "weights/", model_name))
		self.fcn_model.load_state_dict(state_dict)

		self.mask1 = np.zeros((self.h, self.w))
		# self.brand = ['', 'backpack', 'survivor', 'vent', 'phone']
		self.brand = ['', 'phone', 'backpack', 'vent', 'survivor']
		rospy.loginfo("Node ready!")

	
		self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.img_cb)

		self.rgb_pub = rospy.Publisher("FCN_predict_img/", Image, queue_size=1)
		self.mask_pub = rospy.Publisher("FCN_predict_mask/", Image, queue_size=1)

		#self.image_pub = rospy.Publisher("/predict_mask", Image, queue_size=1)
		rospy.loginfo("Start Predicting image")
		self.rgb_data = None


	def img_cb(self, rgb_data):

		self.rgb_data = rgb_data  


		if self.rgb_data is not None:
			cv_image = self.bridge.imgmsg_to_cv2(self.rgb_data, "bgr8")
			cv_image = cv2.resize(cv_image, (640,480), interpolation=cv2.INTER_CUBIC) 
			predict_img, predict_mask, cX, cY = self.predict(cv_image)


			predict_mask[predict_mask > 0] = 255

			self.rgb_pub.publish(
				self.bridge.cv2_to_imgmsg(predict_img, "bgr8"))
			
			self.mask_pub.publish(
				self.bridge.cv2_to_imgmsg(predict_mask, "8UC1"))

			self.rgb_data = None
		


	def predict(self, img):
		rgb_predict = img
		img = img[:, :, ::-1]  # switch to BGR

		img = np.transpose(img, (2, 0, 1)) / 255.
		img[0] -= self.means[0]
		img[1] -= self.means[1]
		img[2] -= self.means[2]

		# convert to tensor
		img = img[np.newaxis, :]
		img = torch.from_numpy(img.copy()).float()
		output = self.fcn_model(img)
		output = output.data.cpu().numpy()

		N, _, h, w = output.shape
		mask = output.transpose(0, 2, 3, 1)
		mask = mask.reshape(-1, self.n_class).argmax(axis=1)
		mask = mask.reshape(N, h, w)[0]

		mask = np.asarray(mask, np.uint8)
		mask_predict = mask
		# =======================Filter=================================
		cnts, _ = cv2.findContours(
			mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

		obj_list = []
		cX = 0
		cY = 0

		for c in cnts:
			M = cv2.moments(c)
			if M["m00"] == 0:
				break
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			area = cv2.contourArea(c)

			if area > 100:# ====Modify====
				print(area)
				cv2.circle(rgb_predict, (cX, cY), 10, (1, 227, 254), -1)
				class_name = mask[cY][cX]
				cv2.putText(rgb_predict, self.brand[mask[cY][cX]], (
					cX-50, cY-40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (252, 197, 5), 3)
				obj_list.append([cX, cY, class_name])

		return rgb_predict, mask_predict, cX, cY

	def onShutdown(self):
		rospy.loginfo("Shutdown.")		


if __name__ == '__main__':
	rospy.init_node('FCN_Pix2Pix_PREDICT')
	foo = FCN_Pix2Pix_PREDICT()
	rospy.on_shutdown(foo.onShutdown)
	rospy.spin()
