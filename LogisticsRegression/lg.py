# Loading the data (cat/non-cat)
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import urllib2
import zipfile

url_utils = 'https://yqall02.baidupcs.com/file/9c206ca31f9236d644df5d045cd52733?bkt=p3-14009c206ca31f9236d644df5d045cd52733b1a976f1000000000370&fid=1444244650-250528-121075026815863&time=1506096334&sign=FDTAXGERLQBHSK-DCb740ccc5511e5e8fedcff06b081203-IR006pRESbwxTbLLIXh8SJTmpG4%3D&to=73&size=880&sta_dx=880&sta_cs=5&sta_ft=py&sta_ct=0&sta_mt=0&fm2=MH,Yangquan,Netizen-anywhere,,beijing,cnc&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=14009c206ca31f9236d644df5d045cd52733b1a976f1000000000370&sl=76480590&expires=8h&rt=sh&r=858111633&mlogid=6146187933770095681&vuk=1444244650&vbdid=2210445204&fin=lr_utils.py&fn=lr_utils.py&rtype=1&iv=0&dp-logid=6146187933770095681&dp-callid=0.1.1&hps=1&tsl=80&csl=80&csign=xs6FEWhfORLsG1PehuXKFuD25vM%3D&so=0&ut=6&uter=4&serv=0&uc=263553720&ic=4098681609&ti=5e666840c78f19731eefa43542a9238b3d342d4b6f9b131a305a5e1275657320&by=themis'
url_data = 'https://nj02all01.baidupcs.com/file/f05ea8282ffec4d58f78497da62821e4?bkt=p3-1400f05ea8282ffec4d58f78497da62821e48f32d0a90000002a8f9a&fid=1444244650-250528-113349925168274&time=1506096403&sign=FDTAXGERLQBHSK-DCb740ccc5511e5e8fedcff06b081203-IAgnDyAqflNgGHz1m%2FFpj64%2B%2FCI%3D&to=69&size=2789274&sta_dx=2789274&sta_cs=1&sta_ft=zip&sta_ct=0&sta_mt=0&fm2=MH,Guangzhou,Netizen-anywhere,,beijing,cnc&newver=1&newfm=1&secfm=1&flow_ver=3&pkey=1400f05ea8282ffec4d58f78497da62821e48f32d0a90000002a8f9a&sl=76480590&expires=8h&rt=sh&r=163828029&mlogid=6146206552172535063&vuk=1444244650&vbdid=2210445204&fin=datasets.zip&fn=datasets.zip&rtype=1&iv=0&dp-logid=6146206552172535063&dp-callid=0.1.1&hps=1&tsl=80&csl=80&csign=xs6FEWhfORLsG1PehuXKFuD25vM%3D&so=0&ut=6&uter=4&serv=0&uc=263553720&ic=4098681609&ti=117d923a205684f92d53e81922c27cde61e27e50cb5dadd0&by=themis'

utils = urllib2.urlopen(url_utils)
data = urllib2.urlopen(url_data)

file_utils = utils.read()
dataset = data.read()

# save data to file
with open("lr_utils.py", "wb") as code:
   code.write(file_utils)

with open("datasets.zip", "wb") as code:
   code.write(dataset)

# unzip datasets
with zipfile.ZipFile("datasets.zip","r") as zip_ref:
    zip_ref.extractall("")

from lr_utils import load_dataset
import skimage
import matplotlib


def load_dataset():
    train_dataset = h5py.File('datasets/data.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/data.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
example = train_set_x_orig[index]
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# https://github.com/andersy005/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week2/Programming-Assignments/Logistic_Regression_with_a_Neural_Network_mindset_v3.ipynb
