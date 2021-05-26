#!/usr/bin/env python3
import sys
import numpy as np
import cv2
import roslib
import rospy
# import tf

import struct
import time
import os
import rospkg
import math
import argparse

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
# from std_msgs.msg import Header
import message_filters
from datetime import datetime

from torchvision import transforms, utils, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.vgg import VGG
# from sklearn.metrics import confusion_matrix
# from subt_msgs.msg import *

import torch.backends.cudnn as cudnn
from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from data.config import update_config, COLORS, cfg
from utils.output_utils import NMS, after_nms, draw_img

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)



class yolact_prediction(object):
	def __init__(self):

		parser = argparse.ArgumentParser(description='YOLACT Predict in ROS')
		parser.add_argument('--visual_top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
		parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
		parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
		parser.add_argument('--hide_bbox', default=True, action='store_true', help='Whether to display bboxes')
		parser.add_argument('--hide_score', default=True, action='store_true', help='Whether to display scores')
		parser.add_argument('--show_lincomb', default=False, action='store_true',
							help='Whether to show the generating process of masks.')
		parser.add_argument('--no_crop', default=False, action='store_true',
							help='Do not crop output masks with the predicted bounding box.')
		parser.add_argument('--real_time', default=True, action='store_true', help='Show the detection results real-timely.')
		parser.add_argument('--visual_thre', default=0.3, type=float,
							help='Detections with a score under this threshold will be removed.')
		self.args = parser.parse_args()

		r = rospkg.RosPack()
		self.bridge = CvBridge()

		self.path = r.get_path('yolact_prediction')
		model_name = "/src/weights/best_89.48_res101_custom_610000.pth"
		strs = model_name.split('_')
		config = strs[-3] + "_" + strs[-2] + "_config"
		update_config(config)
		print("Using " + config + " according to the trained_model.")

		with torch.no_grad():

			self.cuda = torch.cuda.is_available()
			if self.cuda:
				cudnn.benchmark = True
				cudnn.fastest = True
				torch.set_default_tensor_type('torch.cuda.FloatTensor')
			else:
				torch.set_default_tensor_type('torch.FloatTensor')

			self.net = Yolact()
			self.net.load_weights(self.path + model_name, self.cuda)
			print('Model loaded.')

			if self.cuda:
				self.net = self.net.cuda()

			self.time_here = 0
			self.frame_times = MovingAverage()

			#### Publisher
			self.rgb_pub = rospy.Publisher("Yolact_predict_img/", Image, queue_size=1)

			image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size = 1)

			print ("============ Ready ============")

	def img_cb(self, rgb_data):
		
		self.rgb_data = rgb_data  
		
		if self.rgb_data is not None:
			cv_image = self.bridge.imgmsg_to_cv2(self.rgb_data, "bgr8")

			predict_img = self.predict(cv_image)


			self.rgb_pub.publish(
				self.bridge.cv2_to_imgmsg(predict_img, "bgr8"))

		

			self.rgb_data = None


	def predict(self, img):

		rgb_origin = img
		img_numpy = img

		img = torch.from_numpy(img.copy()).float()
		img = img.cuda()

		img_h, img_w = img.shape[0], img.shape[1]
		img_trans = FastBaseTransform()(img.unsqueeze(0))

		net_outs = self.net(img_trans)
		nms_outs = NMS(net_outs, 0)

		results = after_nms(nms_outs, img_h, img_w, crop_masks=not self.args.no_crop, visual_thre=self.args.visual_thre)
		torch.cuda.synchronize()


		temp = self.time_here
		self.time_here = time.time()

	
		self.frame_times.add(self.time_here - temp)
		fps = 1 / self.frame_times.get_avg()

		frame_numpy = draw_img(results, img, self.args, class_color=True, fps=fps)
		
		return frame_numpy


	def onShutdown(self):
		rospy.loginfo("Shutdown.")
		torch.cuda.empty_cache()



if __name__ == '__main__':
	rospy.init_node('yolact_prediction', anonymous=False)
	yolact_prediction = yolact_prediction()
	rospy.on_shutdown(yolact_prediction.onShutdown)
	rospy.spin()
