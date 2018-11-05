import numpy as np

def J(theta ,X_b ,y):
  try:
    return np.sum((y - X_b.dot(theta))**2 )/len(y)
  except:
    return float("inf")

def dJ(theat,X_b,y):
  return X_b.T.dot(X_b.dot(theta) - y) * 2. /len(y)
  
def dJ_sgd(theta . X_b_i , y_i):
  return X_b_i.T.dot(X_b_i.dot(theta) - y_i) *2 

def sgd(X_b ,y initial_theta , n_iters):
  t0 = 5 
  t1 = 50 
  def  learning_rate(t):
    return t0/(t + t1)
  
  theta = initial_theta
  for cur_iter in range(n_iters):
    ran_i = np.random.randint(len(x_b))
    gradient = dJ_sgd(theta ,X_b[rand_i] , y[rand_i])
    theta = theta - learning_rate(cur_iter) * gradient 
  
  return theta 
