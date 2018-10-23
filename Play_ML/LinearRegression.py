import numpy as np 
import matplotlib.pyplot as plt 

class SimpleLinearRegression1:
  
  def __init__(self):
    self.a_ = None 
    self.b_ = None 
    
  def fit(self , x_train , y_train):
    assert x_train.ndim == 1 ,\
    assert len(x_train) == len(y_train),\
    
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    
    num ,d = 0.0,0.0
    for x ,y in zip(x_train ,y_train):
      num += (x - x_mean)*(y-y_mean)
      d += (x-x_mean)**2
    
    self.a_ = num/d
    self.b_ = y_mean = self.a_ * x_mean
    return self 
  def predict(self , x_predict):
  """给定的预测数据集x_predict ,返回便是x_predict的结果向量""""
  assert x_predict.ndim == 1 
  assert self.a_ is not None and self.b_ is not None 
  
  return np.array([self._predict(x) for x in x_predict])
  
  def _predict(self , x_num):
    
    return self.a_ * x_num + self.b_
  def __repr__(self):
    return "SimpleLinearRegression()"
    
class SimpleLinearRegression2:
  """
  向量化运算的速度大约是50倍的使用for循环的方法
  """
  
  def __init__(self):
    self.a_ = None 
    self.b_ = None 
    
  def fit(self , x_train , y_train):
    assert x_train.ndim == 1 ,\
    assert len(x_train) == len(y_train),\
    
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    
#   num ,d = 0.0,0.0
#   for x ,y in zip(x_train ,y_train):
#    num += (x - x_mean)*(y-y_mean)
#    d += (x-x_mean)**2
    num = (x_train - x_mean).dot(y_train - y_mean)
    d = (x_train - x_mean).dot(x_train - x_mean)
    self.a_ = num/d
    self.b_ = y_mean = self.a_ * x_mean
    return self 
  def predict(self , x_predict):
  """给定的预测数据集x_predict ,返回便是x_predict的结果向量""""
  assert x_predict.ndim == 1 
  assert self.a_ is not None and self.b_ is not None 
  
  return np.array([self._predict(x) for x in x_predict])
  
  def _predict(self , x_num):
    
    return self.a_ * x_num + self.b_
  def __repr__(self):
    return "SimpleLinearRegression()"
