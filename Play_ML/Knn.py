import numpy as np 
from math import sqrt
from collections import Counter

class Knn:
  '''
  bulid my own knn 
  '''
  def __init__(self,k):
    """初始化knnn分类器"""
    assert  k >= 1 
    self.k = k
    self._X_train = None
    self._y_train = None
  
  def fit(self , X_train, y_train):
    """根据训练集X_train和y_train训练kNN分类器"""
    assert X_train.shape[0] == y_train.shape[0]
    assert self.k <= X_train.shape[0]
    self._X_train = X_train
    self._y_train = y_train
    return self 
  
  def predict(self ,X_predict):
    """给定带预测的数据集X_predict ，返回X_predict的结果向量"""
    assert self._X_train is not None and self._y_train is not None
    "必须先运行完fit函数"
    assert X_predict.shape[1] == self_X_train.shape[1]
    "特征的个数应该与训练集中的特征个数相同"
    
    y_predict = [self._predict(x) for x in X_predict]
    return np.array(y_predict)
    
  def _predict(x):
    """给定单个待预测的数据x，但会x的预测结果"""
    assert x.shape[0] == self._X_train.shape[1]
    distances =[sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
    nearest = np.argsort(distances)
    
    topK_y = [sefl._y_train[i] for i in nearest[:self.k]]
    votes = Counter(topK_y)
    
    return votes.most_cpmmon(1)[0][0]
  
  def __repr__(self):
    return "KNN(k=%d)"%self.k
