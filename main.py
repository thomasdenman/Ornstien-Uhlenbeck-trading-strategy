import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import scipy.optimize as so
from collections import deque


def data(stock,start,end):
    data = pdr.get_data_yahoo(stocks,start,end)
    data = data['Close']
    data = data.interpolate(method = 'linear',axis = 0)
    returns = data.pct_change()
    meanreturns = returns.mean()
    covmatrix = returns.cov() 
    return data , covmatrix



stocks = ['KO','PEP']

end = dt.datetime.now()
start = end-dt.timedelta(days=1000)

data , covmatrix = data(stocks,start,end)

#print(data.head())
#for stock in stocks:
 #   plt.plot(data[stock])

#plt.imshow(covmatrix)
#plt.show()


def spread(data,beta,stocks):
    spread = data[stocks[0]]+beta*data[stocks[1]]
    return spread

spread1 = spread(data,0.1,stocks)


def blackscholes():
    Time = 1
    mu = 0.01
    S_0 =100
    numsteps = 1000
    sigma = 0.3
    dt = Time/numsteps

    S_t = np.exp((mu-sigma **2 /2)*dt+sigma*np.random.normal(0,np.sqrt(dt),size = [10000,numsteps])).T
    S_t = np.vstack([np.ones(10000),S_t])
    S_t = S_0*np.cumprod(S_t,0)
    return S_t

stock = blackscholes()

def __compute_log_likelihood(params, *args):

    theta, mu, sigma = params
    X, dt = args
    n = len(X)

    sigma_tilde_squared = sigma ** 2 * (1 - np.exp(-2 * mu * dt)) / (2 * mu)
    summation_term = 0

    for i in range(1, len(X)):
        summation_term += (X[i] - X[i - 1] * np.exp(-mu * dt) - theta * (1 - np.exp(-mu * dt))) ** 2

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    log_likelihood = (-np.log(2 * np.pi) / 2) + (-np.log(np.sqrt(sigma_tilde_squared))) + summation_term

    return -log_likelihood



def MLE(X, dt, tol=1e-4):


    bounds = ((None, None), (1e-5, None), (1e-5, None))  # theta ∈ ℝ, mu > 0, sigma > 0
                                                           # we need 1e-10 b/c scipy bounds are inclusive of 0,
                                                           # and sigma = 0 causes division by 0 error
    theta_init = np.mean(X)
    initial_guess = (theta_init, 100, 100)  # initial guesses for theta, mu, sigma
    result = so.minimize(__compute_log_likelihood, initial_guess, args=(X, dt), bounds=bounds)
    theta, mu, sigma = result.x
    max_log_likelihood = -result.fun  # undo negation from __compute_log_likelihood
    # .x gets the optimized parameters, .fun gets the optimized value
    return theta, mu, sigma, max_log_likelihood

theta , mu,sigma, max_log_likelihood = estimate_coefficients_MLE(spread1,1/691,tol = 1e-4 )

def UO(theta,mu,sigma):
    dt = 1/691
    n =691
    x=np.zeros(n)
    x[0]= spread1[0]
    for i in range(0,n-1):
        x[i+1] = x[i]+mu*(theta-x[i])*dt+sigma*np.sqrt(dt)*np.random.randn()
    return x
simulated = np.zeros([691,100])
for i in range(0,100):
    simulated[:,i] = UO(theta,mu,sigma)
def t_optimal(c):
    t_hat = 0.5*np.log(1+0.5*np.sqrt((c**2-3)**2+4*c**2)+c**2-3)
    T_star = 1/mu*t_hat
    return T_star

plt.plot(simulated)
plt.plot(np.array(spread1))

plt.show()


class Portfolio:
    '''
   portfolio of holding $1 of stock A and -$alloc_B of stock B
    '''

    def __init__(self, price_1, price_2, alpha):
        self.initalprice_1 = price_1
        self.initalprice_2 = price_2
        self.currentprice_1 = price_1
        self.currentprice_2 = price_2
        self.alpha = alpha

    def Update(self, newprice_1, newprice_2):
        self.currentprice_1 = newprice_1
        self.currentprice_2 = newprice_2

    def Value(self):
        return self.currentprice_1 / self.initialprice_1 - self.alpha * self.currentprice_2 / self.initalprice_2



class Model:
    def __init__(self):

        self.alpha = -1

        self.time = deque(maxlen=200)  # RW's aren't supported for datetimes
        self.close_1 = RollingWindow[float](200)
        self.close_2 = RollingWindow[float](200)

        self.portfolio = None

    def Update(self, time, close_1, close_2):
        '''
        Adds a new point of data to our model, which will be used in the future for training/retraining
        '''
        if self.portfolio is not None:
            self.portfolio.Update(close_1, close_2)

        self.time.appendleft(time)
        self.close_1.Add(close_1)
        self.close_2.Add(close_2)

    def portfoliovalue(self, ts_A, ts_B, alpha):
        ts_A = ts_A.copy()
        ts_B = ts_B.copy()
        ts_A = ts_A / ts_A[0]
        ts_B = ts_B / ts_B[0]
        return ts_A - alpha * ts_B

    def maxalpha(self, ts_A, ts_B, dt):

        theta = mu = sigma = alpha = 0
        max_log_likelihood = 0

        def compute_coefficients(x):
            portfolio_values = self.portfoliovalue(ts_A, ts_B, x)
            return MLE(portfolio_values, dt)

        vectorized = np.vectorize(compute_coefficients)
        linspace = np.linspace(.01, 1, 100)
        res = vectorized(linspace)
        index = res[3].argmax()
        return res[0][index], res[1][index], res[2][index], linspace[index]

    def EnterTrade(self):
        if self.portfolio.Value() > self.theta + self.theta*self.mu/np.sqrt(self.sigma):

            return True
        if self.portfolio.Value() < self.theta + self.theta*self.mu/np.sqrt(self.sigma):
            return True

    def ExitTrade(self,entrytime,time):
        if time >= entrytime+self.T_star:
            return True






    def Train(self):
        '''
        Computes our OU and alpha coefficients
        '''

        # remember RollingWindow time order is reversed
        ts_A = np.array(list(self.close_1))[::-1]
        ts_B = np.array(list(self.close_2))[::-1]

        days = (self.time[0] - self.time[-1]).days
        dt = 1.0 / days

        self.theta, self.mu, self.sigma, self.alpha = self.maxalpha(ts_A, ts_B, dt)

        self.T_star = t_optimal(self.theta)

        self.portfolio = Portfolio(ts_A[-1], ts_B[-1], self.alpha)


