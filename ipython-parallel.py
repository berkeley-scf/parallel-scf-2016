import numpy as np
np.random.seed(0)
n = 500
p = 50
X = np.random.normal(0, 1, size = (n, p))
Y = X[: , 0] + pow(abs(X[:,1] * X[:,2]), 0.5) + X[:,1] - X[:,2] + np.random.normal(0, 1, n)

def looFit(index, Ylocal, Xlocal):
    rf = rfr(n_estimators=100)
    fitted = rf.fit(np.delete(Xlocal, index, axis = 0), np.delete(Ylocal, index))
    pred = rf.predict(np.array([Xlocal[index, :]]))
    return(pred[0])

from ipyparallel import Client
c = Client()
c.ids

dview = c[:]
dview.block = True
dview.apply(lambda : "Hello, World")

lview = c.load_balanced_view()
lview.block = True

dview.execute('from sklearn.ensemble import RandomForestRegressor as rfr')
dview.execute('import numpy as np')
mydict = dict(X = X, Y = Y, looFit = looFit)
dview.push(mydict)

nSub = 50  # for illustration only do a subset

# need a wrapper function because map() only operates on one argument
def wrapper(i):
    return(looFit(i, Y, X))

import time
time.time()
pred = lview.map(wrapper, range(nSub))
time.time()

print(pred[0:10])

# import pylab
# import matplotlib.pyplot as plt
# plt.plot(Y, pred, '.')
# pylab.show()
