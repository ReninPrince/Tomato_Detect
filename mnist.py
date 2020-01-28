import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
df_train  = pd.read_csv('../input/train.csv')
