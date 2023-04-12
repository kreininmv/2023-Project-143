#@title Класс логистической регрессии
import numpy as np
import scipy.linalg as la
import random
from sklearn.metrics import accuracy_score
from datetime import datetime as dt
from tqdm import tqdm

class MyLogregression:
    def __init__(self, fit_intercept=False, eps=5*1e-3,     iter =10 ** 3, batch=False, batch_size=50, decreasing_lr=False, name="GD", lr_func = None, label="None", dependent=False, l2_coef=0, betas = [0.999, 0.99]):
        self.fit_intercept = fit_intercept
        self._iter         = iter
        self._batch        = batch
        self._eps          = eps
        self._opt          = self.choose_opt_method(name)
        self._name         = name
        self._batch_size   = batch_size
        self._decreasing_lr = decreasing_lr
        self._lr_func      = lr_func 
        self._label        = label
        self._dependent    = dependent
        self._l2_coef      = l2_coef
        self._betas        = betas
        self._grad_function = self.__grad_function
        if name == "AdamL2" or name == "AdamW":
            self._error_criterion = lambda w: np.linalg.norm(self._grad_function(w) + self._l2_coef * w, 2)
        else:
            self._error_criterion = lambda w: np.linalg.norm(self._grad_function(w), 2)
        self._to_seconds = lambda s: s.microseconds * 1e-6 + s.seconds
    
    def __function(self, w):
        sum = 0
        n = self._X_train.shape[0]

        for i in range(len(self._y_train)):     
            sum = sum + 1/n * np.log(1 + np.exp(-self._y_train[i] * self._X_train[i, :] @ w)) 
        
        return sum
    
    def __grad_function(self, w):
        sum = np.zeros(w.shape)
        n = self._X_train.shape[0]
        
        for i in range(len(self._y_train)):            
            up = self._y_train[i] * self._X_train[i] * np.exp(-self._y_train[i] * w * self._X_train[i])
            down = n * (1 + np.exp(-self._y_train[i] * w * self._X_train[i]))
            sum = sum  - up/down

        return sum

    def __add_constant_column(self, X):
        n, k = X.shape
        if self.fit_intercept:
            X_train = np.hstack((X, np.ones((n, 1))))
            #w0 = np.random.rand(k+1)
            w0 = np.zeros(k+1)
        else:
            X_train = X
            #w0 = np.random.rand(k)
            w0 = np.zeros(k)
        return X_train, w0
    
    def fit(self, X, y, X_test, y_test):
        n, k = X.shape
        wo = np.array([])
        X_train, w0 = self.__add_constant_column(X)
        self._X_train = X_train
        self._y_train = y
        self._X_test = X_test
        self._y_test = y_test

        if self._lr_func is None:
            hessian = 2 / n * X_train.T @ X_train
            wb, vb = np.linalg.eigh(hessian)
            self._lr_func = lambda w: 1/wb[-1]
        
        #self._error_criterion = lambda X, y, w, n: np.linalg.norm(self.__grad_function(X, y, w, n), 2)
        #self._grad_function = lambda w: 2/n * X_train.T @ (X_train @ w - y)
        
        
        # Initialize variables
        self._errors     = []
        self._error_adamw = []
        self._accuracies = []
        self._w          = w0
        self._time       = []
        self._i          = 0

        self._w = self._opt(self.__function, self._grad_function, self._w, self._lr_func)
        return 

    def predict(self, X):
        return self._w @ X.T

    def get_weights(self): return self._w
    
    def get_errors(self): return self._errors

    def get_accuracy(self): return self._accuracies

    def get_weights(self): return self._w

    def get_time(self): return self._time

    def get_name(self): return self._label
  
    def __gradient_descent(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        for k in range(self._iter):
            self._i += 1
            self._w = self._w - lr(0) * grad_f(self._w)

            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
        return self._w
    
    def __AdamL2(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        
        grad = grad_f(self._w) + self._l2_coef * self._w
        m = grad
        v = grad * grad
        eps = 1e-8

        for k in range(self._iter):
            self._i += 1
            grad = grad_f(self._w) + self._l2_coef * self._w
            
            m = self._betas[0] * m + (1-self._betas[0]) * grad
            v = self._betas[1] * v + (1-self._betas[1]) * (grad * grad)
            
            m_bias_corr = m
            v_bias_corr = v
            #m_bias_corr = m / (1-self._betas[0] ** self._i)
            #v_bias_corr = v / (1-self._betas[1] ** self._i)

            self._w = self._w - lr(0) * grad_f(self._w)
            self._w = self._w - lr(0) * m_bias_corr / (np.sqrt(v_bias_corr) + eps)
            
            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w

    def __AdamW(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        grad = grad_f(self._w)
        m = grad
        v = grad * grad
        eps = 1e-8
        self._i = 0
        for k in range(self._iter):
            self._i += 1
            grad = grad_f(self._w)
            
            m = self._betas[0] * m + (1-self._betas[0]) * grad
            v = self._betas[1] * v + (1-self._betas[1]) * (grad * grad)
            
            #m_bias_corr = m / (1-self._betas[0] ** self._i)
            #v_bias_corr = v / (1-self._betas[1] ** self._i)
            m_bias_corr = m 
            v_bias_corr = v

            self._w = self._w - lr(0) * grad_f(self._w)
            self._w = self._w - lr(0) * m_bias_corr / (np.sqrt(v_bias_corr) + eps) - lr(0) * self._l2_coef * self._w
            
            self._error_adamw.append(np.linalg.norm(m_bias_corr + self._l2_coef * self._w * v_bias_corr , 2))
            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w
    
    def __MyAdamW(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        grad = grad_f(self._w)
        m = grad
        v = grad * grad
        eps = 1e-8
        for k in range(self._iter):
            self._i += 1
            grad = grad_f(self._w) + self._l2_coef * self._w
            
            m = self._betas[0] * m + (1-self._betas[0]) * grad
            v = self._betas[1] * v + (1-self._betas[1]) * (grad * grad)
            
            m_bias_corr = (m +  self._l2_coef * self._w) / (1-self._betas[0] ** self._i)

            v_bias_corr = v / (1-self._betas[1] ** self._i)
            #v_bias_corr = max(v / (1-self._betas[1] ** self._i), self._l2_coef)

            self._w = self._w - lr(0) * grad_f(self._w)
            self._w = self._w - (self._i + 1)/(self._i + 4)*lr(0) * m_bias_corr / (np.sqrt(v_bias_corr) + eps) 

            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
            
        return self._w
    
    def __OASIS(self, f, grad_f, w0, lr):
        time_start = dt.now()
        self._w = w0
        n = self._X_train.shape[0]
        grad = grad_f(self._w)
        m = grad
        hessian = 0
        for x, y in zip(self._X_train, self._y_train):
            hessian += 1/n * x * x * np.exp(-self._w @ x * y) / ((1 + np.exp(-self._w @ x * y)) ** 2)
        v = hessian

        lr_k = lr(0)
        lr_prev_k = lr(0) 
        lr_relative = 1

        eps = 1e-8
        w_prev = np.zeros_like(w0)
        grad_prev = np.ones_like(w0)

        for k in range(self._iter):            
            self._i += 1
            # Запоминаем градиент и обновляем новый
            grad_prev = grad
            grad = grad_f(self._w) + self._l2_coef * self._w
            
            # Считаем матрицу D_K
            D_k = np.ones_like(grad)
            for r in range(self._y_train.size):
                x = self._X_train[r, :]
                y = self._y_train[r]
                D_k += 1/n * x * x * np.exp(-self._w @ x * y) / ((1 + np.exp(-self._w @ x * y)) ** 2)
        
            # Делаем среднезвешенное
            m = self._betas[0] * m + (1-self._betas[0]) * grad
            v = self._betas[1] * v + (1-self._betas[1]) * D_k
            

            # Делаем bias correction
            m_bias_corr = m/ (1-self._betas[0] ** self._i)
            v_bias_corr = v / (1-self._betas[1] ** self._i)

            # Обновляем learning rate
            left = np.sqrt(1 + lr_relative) * lr_k
            right = np.linalg.norm((self._w - w_prev) * D_k, 2) / np.max([2 * np.linalg.norm((grad - grad_prev) * D_k, 2), 1e-5])
            lr_k = np.min([left, right, 0.09])
            
            w_prev = self._w
            self._w = self._w - lr_k * m_bias_corr / (np.sqrt(v_bias_corr) + eps)  - lr_k * self._l2_coef * self._w
            
            lr_relative = lr_k / lr_prev_k
            lr_prev_k = lr_k
            self._append_errors(self._w, time_start)
            
            if (self._errors[-1] < self._eps):
                return self._w
            
        return self._w
    
    def _append_errors(self, w, time_start):
        # Tecnichal staff 
        error = self._error_criterion(self._w)
                
        self._time.append(self._to_seconds(dt.now() - time_start))
        self._errors.append(error) 
        
        answer = self.predict(self._X_test)
        answer = np.sign(answer)
    
        self._accuracies.append(accuracy_score(self._y_test, answer))
    
    def choose_opt_method(self, name):
        if (name == 'AdamL2'):
            return self.__AdamL2
        if (name == 'AdamW'):
            return self.__AdamW
        if (name == 'MyAdamW'):
            return self.__MyAdamW
        if (name == 'OASIS'):
            return self.__OASIS
        return self.__gradient_descent
