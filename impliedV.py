import numpy as np
from scipy.stats import norm


class Stock:
    def __init__(self, S):
        """

        :param S: stock price
        """
        self._stockprice = S

    @property
    def S(self):
        return self._stockprice

    @S.setter
    def S(self, new_stockprice):
        self._stockprice = new_stockprice

    def __repr__(self):
        return '<Stock with price {}>'.format(self._stockprice)


class InterestRate:
    def __init__(self, r):
        """

        :param r: interest rate
        """
        self._interestrate = r

    @property
    def r(self):
        return self._interestrate

    @r.setter
    def r(self, new_r):
        self._interestrate = new_r

    def __repr__(self):
        return '<Interest rate is {}%>'.format(self._interestrate)


class CallOption:
    def __init__(self, C, K, t, T, Stock, sigma=0.5, q=0.0):
        """

        :param C: Call option price
        :param t:
        :param T:
        :param sigma:
        :param Stock:
        """
        assert t >= 0, 'starting time cannot be negative'
        assert C >= 0, 'Call option price cannot be negative'
        # assert K >= Stock.S, 'strike price should be larger than current stock price'
        self._C = C
        self._t = t
        self._K = K
        self._T = T
        self._sigma = sigma
        self._Stock = Stock
        self._q = q

    @property
    def C(self):
        print("Call Option Price is {}".format(self._C))
        return self._C

    @C.setter
    def C(self, calloption_price):
        assert calloption_price >= 0, 'call option price cannot be smaller than 0'
        self._C = value

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, strike_price):
        assert strike_price > self.Stock.S, 'Strike Price of Call Option should not be smaller than Current Stock Price'
        self._K = strike_price

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, start_time):
        assert start_time >=0, 'start time t cannot be smaller than zero'
        self._t = start_time

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, expiration_time):
        self._T = expiration_time

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma_v):
        self._sigma = sigma_v


class ImpliedVolatility(object):
    def __init__(self, Stock, InterestRate, CallOption):
        self._stock = Stock
        self._InterestRate = InterestRate
        self._CallOption = CallOption

    def impVol_newton(self, max_iteration=100, precision=1.0e-5):
        self._CallOption.sigma = 0.5
        for step in range(max_iteration):
            diff = self._CallOption.C - self.bsm()
            print(diff)
            if abs(diff) < precision:
                return self._CallOption.sigma
            self._CallOption.sigma = self._CallOption.sigma + diff / self.vega()
        print("Not converged")
        return self._CallOption.sigma

    def bsm(self, q=0.0):
        S = self._stock.S
        K = self._CallOption.K
        t = self._CallOption.t
        T = self._CallOption.T
        r = self._InterestRate.r
        d1 = self.d1()
        d2 = self.d2()
        return S*np.exp(-q*(T - t))*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)

    def vega(self, q=0.0):
        S = self._stock.S
        K = self._CallOption.K
        t = self._CallOption.t
        T = self._CallOption.T
        r = self._InterestRate.r
        d1 = self.d1()
        return S*np.exp(-q*(T - t)) * norm.pdf(d1, 0, 1) * np.sqrt(T - t)

    def d1(self, q=0.0):
        S = self._stock.S
        K = self._CallOption.K
        t = self._CallOption.t
        T = self._CallOption.T
        r = self._InterestRate.r
        sigma = self._CallOption.sigma
        return (np.log(S/K) + (r - q + sigma*sigma/2)*(T-t)) / (sigma * np.sqrt(T-t))

    def d2(self):
        return self.d1() - self._CallOption.sigma * np.sqrt(self._CallOption.T-self._CallOption.t)


if __name__ == '__main__':
    import datetime
    import time
    T = (datetime.date(2014, 10, 18) - datetime.date(2014, 9, 8)).days / 365
    stock = Stock(595)
    interestrate = InterestRate(0.0002)
    c = CallOption(17.5, 585, 0, T, stock)
    t1 = time.time()
    im=ImpliedVolatility(stock, interestrate, c)
    print(im.impVol_newton())
    print("time", time.time()-t1)
