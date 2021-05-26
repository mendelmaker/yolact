import os
import sys
import logging
import gdown
from zipfile import ZipFile

rosbag_name = 'xavier-predict.bag'
rosbag_url = 'https://drive.google.com/u/2/uc?id=10qqGV1O-AhdgfW7CEX4As9L63f046439&export=download'

if not os.path.isfile(rosbag_name):
    
    gdown.download(rosbag_url, output=rosbag_name, quiet=False)

print("Finished downloading rosbag.") 