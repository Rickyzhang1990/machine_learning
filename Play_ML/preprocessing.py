import numpy as np 


class StandardScaler:
  def __init__(self):
    self.mean_ = None 
    self.scale_ = None
    
  def fit(self , X):
    assert X.ndim == 2;"只处理2维数组"
    self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
    self.std_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
    return self   
  def transform(self , x):
    assert x.ndim == 2 
    assert self.mean_ is not None and self.std_ is not None 
    res_x = np.empty(shape = x.shape ,dtype = float)
    for col in range(x.shape[1]):
      res_x[:,col] = (x[:,col] - self.mean_[:,col])/self.std_[:,col]
    return res_x
    
    
class MinMaxScaler:
  
  def __init__(self):
    self.min_ = None 
    self.max_ = None 
  
  def fit(self,X):
    assert  X.ndim == 2 """只处理二维数组"""
    self.min_ = np.array([np.min(X[:,i] for i in range(X.shape[1])])
    self.max_ = np.array([np.max(X[:,i] for i in range(X.shape[1])])
    return self 
                          
  def transform(self , x):
    assert x.shape[1] == self.min_.shape[1];"""输入检验数据集的特征数与最大值最小值的数目相等"""                     
    assert self.min_ is not None and self.max_ is not None;"""最大值最小值矩阵经过fit过程，不为空值""" 
    res_x = np.empty(shape = x.shape ,dtype = float)
    for col in range(x.shape[1]):
      res_x[:,col] = (x[:,col] - self.min_[:,col])/(self.max_[:,col] - self.min_[:,col])
    return rex_x
