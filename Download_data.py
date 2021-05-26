import os
import sys
import logging
import gdown
from zipfile import ZipFile

data_url = 'https://drive.google.com/a/g2.nctu.edu.tw/uc?id=1gADkkutnJ3Qn3Rx_6br9pkX8yn-JJsDi&export=download'
data_name = 'data'
gdown.download(data_url, output=data_name + '.zip', quiet=False)
zip1 = ZipFile(data_name + '.zip')
zip1.extractall(data_name)
zip1.close()
os.remove(data_name + ".zip")

print("Finished downloading data.") 