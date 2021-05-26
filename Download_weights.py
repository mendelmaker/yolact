import os
import sys
import logging
import gdown
from zipfile import ZipFile

models_url = 'https://drive.google.com/a/g2.nctu.edu.tw/uc?id=1Z3Vx-0BNJ4M6nb4ie40n76Tp5UGZRcgl&export=download'
models_name = 'weights'
if not os.path.isdir(models_name):
    gdown.download(models_url, output=models_name + '.zip', quiet=False)
    zip1 = ZipFile(models_name + '.zip')
    zip1.extractall(models_name)
    zip1.close()
    os.remove(models_name + ".zip")

print("Finished downloading models.") 