import os
import sys
import logging
import gdown
from zipfile import ZipFile

result_url = 'https://drive.google.com/a/g2.nctu.edu.tw/uc?id=1c3ocyO-gDmv-OstM0XbBQdczWGWCANu3&export=download'
result_name = 'results'
if not os.path.isdir(result_name):
    gdown.download(result_url, output=result_name + '.zip', quiet=False)
    zip1 = ZipFile(result_name + '.zip')
    zip1.extractall(result_name)
    zip1.close()
    os.remove(result_name + ".zip")

print("Finished downloading results.") 