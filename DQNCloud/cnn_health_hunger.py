import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
import os
from PIL import Image
import torchvision.transforms as T
from config import CONVOLUTION_WIDTH, CONVOLUTION_HEIGHT, TRAINING_DATA, DEVICE, CNN_STATE_DICT_PATH, RESOLUTION_WIDTH, RESOLUTION_HEIGHT
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

class CNNHealthHunger(nn.Module):
  
  resize = T.Compose([T.Resize(CONVOLUTION_HEIGHT, interpolation=Image.CUBIC)])
  train_x = None
  train_y = None
  test_x = None
  test_y = None
  to_tensor = T.Compose([T.ToTensor()])

  def __init__(self):
      super(CNNHealthHunger, self).__init__()
      linear_input_size = 4 * 2 * 7
      self.cnn_layers = nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Defining another 2D convolution layer
        nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
      )

      self.linear_layers = nn.Sequential(
          nn.Linear(linear_input_size, 2)
      )

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layers(x)
    return x


  def load_training_data(self, training_type, training_class1_path, training_class2_path, test_class1_path, test_class2_path):
    
    train_class1 = os.listdir(training_class1_path)
    train_class2 = os.listdir(training_class2_path)
    test_class1 = os.listdir(test_class1_path)
    test_class2 = os.listdir(test_class2_path)
    
    train_data = {
      "image": [],
      "label": []
    }

    test_data = {
      "image": [],
      "label": []
    }

    for class1 in train_class1:
      s_im = self.process_image_test(os.path.join(training_class1_path,class1),training_type)
      train_data["image"].append(s_im)
      train_data["label"].append(1)
    
    for class2 in train_class2:
      s_im = self.process_image_test(os.path.join(training_class2_path,class2), training_type)
      train_data["image"].append(s_im)
      train_data["label"].append(0)

    for class1 in test_class1:
      s_im = self.process_image_test(os.path.join(test_class1_path,class1), training_type)
      test_data["image"].append(s_im)
      test_data["label"].append(1)
    
    for class2 in test_class2:
      s_im = self.process_image_test(os.path.join(test_class2_path,class2), training_type)
      test_data["image"].append(s_im)
      test_data["label"].append(0)
    
    self.train_x = train_data["image"]
    self.train_y = np.array(train_data["label"]).astype(float)
    self.test_x = test_data["image"]
    self.test_y = np.array(test_data["label"]).astype(float)  
    self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train_x, self.train_y, test_size = 0.1)
  
  def process_image_test(self, img_path, crop_type = "health"):
    im = Image.open(img_path)
    r_im = self.process_image(im)
    a_im = self.crop_image(r_im, crop_type)
    return a_im

  def crop_image(self, r_im, crop_type = "health"):
    w,h = (RESOLUTION_WIDTH, RESOLUTION_HEIGHT)
    rw,rh = r_im.size
    wf = rw/w
    hf = rh/h
    if crop_type == 'health':
      x = 218
      y = 522
      x2 = 162
      y2 = 20
    else:
      x = 420
      y = 522
      x2 = 162
      y2 = 20

    c_im = T.functional.crop(r_im, int(y*hf),int(x*wf), int(y2*hf),int(x2*wf))
    a_im = np.array(c_im).astype(float).transpose((2, 0, 1))
    return a_im

  def process_image(self, im):
    return self.resize(im)

def train_model(epochs, model, save_path):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
      optimizer = Adam(model.parameters(), lr=0.07)
      criterion = nn.CrossEntropyLoss()
      model.train()
      tr_loss = 0
      x_train = Variable(torch.tensor(model.train_x)).float()
      y_train = Variable(torch.tensor(model.train_y)).float()
      x_val = Variable(torch.tensor(model.val_x)).float()
      y_val = Variable(torch.tensor(model.val_y)).float()
      optimizer.zero_grad()
      output_train = model(x_train)
      output_val = model(x_val)
      loss_train = criterion(output_train, y_train.squeeze().long())
      loss_val = criterion(output_val, y_val.squeeze().long())
      train_losses.append(loss_train)
      val_losses.append(loss_val)
      loss_train.backward()
      optimizer.step()
      tr_loss = loss_train.item()
      if epoch%2 == 0:
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
      # plotting the training and validation loss
    start_time = time.time()
    print('Predicting training set')
    predict_set(x_train, y_train, model)
    end_time = time.time()
    print('Prediction of ' + str(len(x_train)) + ' samples took ' + str(end_time - start_time) + ' seconds')
    print('Predicting validation set')
    predict_set(x_val, y_val, model) 
    test_x = Variable(torch.tensor(model.test_x)).float()
    test_y = Variable(torch.tensor(model.test_y)).float()
    print('Predicting test set')
    predict_set(test_x, test_y, model)
    torch.save(model.state_dict(), save_path)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.show()


def predict_set(set_values, labels, model):
    predictions = is_done(set_values, model)
    print(str(accuracy_score(labels, predictions)*100) + '% accuracy on target dataset')

def detect_is_done(np_value, model):
    value = Variable(torch.tensor([np_value]).unsqueeze(1)).float()
    return bool(is_done(value, model)[0])

def is_done(value, model):
    with torch.no_grad():
        output = model(value)

    softmax = torch.exp(output)
    prob = list(softmax.numpy())
    return np.argmax(prob, axis=1)

def train_detector(dataset,class1_folder,class2_folder, model = None, epochs=500):
  if model == None:
    model = CNNHealthHunger()
  save_path = os.path.join(CNN_STATE_DICT_PATH, dataset + '_' + class1_folder + '_' + class2_folder + '_' + 'params.dat')
  dataset_path = os.path.join(TRAINING_DATA, dataset)
  training_class1_dataset = os.path.join(dataset_path, 'train\\' + class1_folder)
  training_class2_dataset = os.path.join(dataset_path, 'train\\' + class2_folder)
  test_class1_dataset = os.path.join(dataset_path, 'test\\' + class1_folder)
  test_class2_dataset = os.path.join(dataset_path, 'test\\' + class2_folder)
  model.load_training_data(dataset, training_class1_dataset, training_class2_dataset, test_class1_dataset, test_class2_dataset)
  train_model(epochs, model, save_path)

train_detector('health', 'healthy', 'unhealthy')
train_detector('health', 'pain', 'no_pain')
train_detector('hunger', 'full', 'not_full')
train_detector('hunger', 'starving', 'fed')
