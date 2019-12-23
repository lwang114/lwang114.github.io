import numpy as np
import csv
import matplotlib.pyplot as plt

DEBUG = False
def readData(datafile):
  X = []
  y = []
  fp = open(datafile, 'r')
  data = csv.reader(fp, delimiter=',')
  for datum in data:
    y.append(2 * (datum[1] == 'M') - 1)
    X.append([float(x) for x in datum[2:]])
  fp.close()
  if DEBUG:
    print(np.array(X).shape, np.array(y).shape)
  return np.array(X), np.array(y)

def calcGrad(X, y, p, lbda, t):
  m, n = X.shape[0], X.shape[1]
  w, b, eps = p[:n], p[n], p[n+1:]
  g = np.zeros((n+m+1, ))
  d = y * (X @ w + b) + eps - 1

  g[:n] = 2 * t * lbda * w - y * X.T @ (1 / d)
  g[n] = - y @ (1 / d)
  g[n+1:] = t / m * np.ones((m,)) - 1 / d - 1 / eps
  return g

def calcHessian(X, y, p, lbda, t, return_grad=False):
  m, n = X.shape[0], X.shape[1]
  w, b, eps = p[:n], p[n], p[n+1:]
  H = np.zeros((n+m+1, n+m+1))
  d = y * (X @ w + b) + eps - 1
  
  # d2F/d2w
  yX = y * X.T
  y2X = y ** 2 * X.T 
  H[:n, :n] = 2 * t * lbda * np.eye(n) + (yX / d) @ (yX / d).T
  # d2F/d2b
  H[n, n] = (y / d) @ (y / d)   
  # d2F/d2e
  H[n+1:, n+1:] = np.eye(m) * (1 / d ** 2 + 1 / eps ** 2)   
  # Cross terms
  H[:n, n] = y2X @ (1 / d ** 2)
  H[n, :n] = H[:n, n].T
  H[:n, n+1:] = yX / (d ** 2)
  H[n+1:, :n] = H[:n, n+1:].T
  H[n+1:, n] = y / (d ** 2)
  H[n, n+1:] = H[n+1:, n].T
  
  return H 

def NewtonMethod(X, y, p0, err_tol=0.01, damp=False, lbda=0.1, t=1):
  p_opt = p0   
  newton_dec = err_tol + 1.
  while newton_dec > err_tol:
    # Compute the gradient w.r.t to w, b, eps  
    g = calcGrad(X, y, p_opt, lbda=lbda, t=t)
    H = calcHessian(X, y, p_opt, lbda=lbda, t=t)
    H_inv = np.linalg.inv(H)    
    newton_step = H_inv @ g
    newton_dec = (g.T @ H_inv @ g) ** .5
    if DEBUG:
      print('newton_dec: ', newton_dec)
    
    if damp:
      p_opt = p_opt - 1 / (1 + newton_dec) * newton_step
    else:
      p_opt = p_opt - newton_step
  
  return p_opt, newton_dec

def firstOrderOracle(X, y, p, lbda):
  m, n = X.shape
  g = np.zeros((n+1,))
  w, b = p[:n], p[n]
  d = (1 - y * (X @ w + b) > 0)
  g[:n] = 2 * lbda * p[:n] - 1 / m * y * X.T @ d
  g[n] = - 1 / m * y @ d 
  return g

def InteriorPointMethod(X, y, lbda, w0, b0, T_max):
  m, n = X.shape[0], X.shape[1] 
  # Path following parameters
  t = 1.
  barrier_nu = 2 * m
  step_gamma = 1.
  # SVM parameters
  w_opt = w0
  b_opt = b0
  eps_opt = np.maximum(1. - y * (X @ w0 + b0), 0.) + 0.1
  f_hist = np.zeros((T_max,))
  
  p_opt = np.zeros((m + n + 1,))   
  p_opt[:n] = w_opt
  p_opt[n] = b_opt
  p_opt[n+1:] = eps_opt    
    
  p_opt, newton_dec = NewtonMethod(X, y, p_opt, err_tol=1/4, damp=True, lbda=lbda, t=t)
  # Run number of T_max iterations
  for i in range(T_max):
    # Apply damped Newton's method to find good initial value for w, b, eps
    # Apply Newton's method to find the optimal solution to F_t
    p_opt, newton_dec = NewtonMethod(X, y, p_opt, lbda=lbda, t=t)
    
    w_opt, b_opt, eps_opt = p_opt[:n], p_opt[n], p_opt[n+1:]
    f_hist[i] = lbda * w_opt.T @ w_opt + np.mean(np.maximum(1 - y * (X @ w_opt + b_opt), 0))  
    print('Iteration %d, Loss %d' % (i, f_hist[i]))
    print('Classification accuracy: ', np.mean(y * (X @ w_opt + b_opt) >= 1))  

    # Update t
    t = (1 + step_gamma / (barrier_nu ** .5)) * t

  print('Final classification accuracy: ', np.mean(y * (X @ w_opt + b_opt) >= 1))  
  return w_opt, b_opt, f_hist

def EllipsoidMethod(X, y, lbda, w0, b0, T_max):
  m, n = X.shape[0], X.shape[1]
  w_opt, b_opt = w0, b0
  
  R = 10.
  Q = R * np.eye(n + 1)
  f_hist = []
  print('Initial classification accuracy: ', np.mean(y * (X @ w_opt + b_opt) >= 1))
  print('Initial classification scores: ', (y * (X @ w_opt + b_opt))[:10])
  
  assert T_max > 0
  c = np.zeros((n + 1,)) 
  c[:n] = w_opt
  c[n] = b_opt
  for i in range(T_max):
    #  print('L2 norm of c: ', (c[:n] @ c[:n]) ** 1/2)
    omega = firstOrderOracle(X, y, c, lbda)
    
    qnorm = (omega @ Q @ omega) ** 0.5
    q = Q @ omega / qnorm

    c = c - 1 / (n + 2) * q 
    Q = (n + 1) ** 2 / ((n + 1) ** 2 - 1) * (Q - 2 / (n + 2) * q.reshape(-1, 1) @ q.reshape(1, -1))
    
    f = lbda * c[:n] @ c[:n] + np.mean(np.maximum(1 - y * (X @ c[:n] + c[n]), 0))
    
    if len(f_hist) == 0:
      w_opt = c[:n]
      b_opt = c[n]
    elif f < min(f_hist):
      w_opt = c[:n]
      b_opt = c[n]
    f_hist.append(lbda * w_opt.T @ w_opt + np.mean(np.maximum(1 - y * (X @ w_opt + b_opt), 0)))
    

    print('Iteration %d, Loss %0.5f' % (i, f_hist[i]))
    print('Upper bound on the distance to optimum', qnorm)
    print('Classification accuracy: ', np.mean(y * (X @ w_opt + b_opt) >= 1))  
 
  print('Final classification accuracy: ', np.mean(y * (X @ w_opt + b_opt) >= 1))
  return w_opt, b_opt, f_hist

if __name__ == '__main__':
  X, y = readData('dataset.csv') 
  m, n = X.shape[0], X.shape[1]
  T_max = 200
  lbda = 1.
  w0 = np.zeros((n,))
  b0 = 0.
  
  w_opt, b_opt, f_hist_ellipsoid = EllipsoidMethod(X, y, lbda, w0, b0, T_max)
  w_opt, b_opt, f_hist_ipm = InteriorPointMethod(X, y, lbda, w0, b0, T_max)

  plt.figure()
  plt.plot(np.arange(T_max), f_hist_ellipsoid, linestyle='--')
  plt.plot(np.arange(T_max), f_hist_ipm, linestyle=':')
  plt.xlabel('Number of iterations')
  plt.ylabel('Objective value')
  plt.legend({'Ellipsoid Method', 'Interior Point Method'}, loc='best')
  plt.show()
