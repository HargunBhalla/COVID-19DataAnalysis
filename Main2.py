import pandas as pd 

data = pd.read_csv("data.csv") 
tc = data['totale_casi']
tt = data['tamponi']
y = []
tt_increase = []
for i in range(1, len(tt)):
    current_epi = (tc[i] - tc[i-1])/(tt[i]-tt[i-1])*100
    tt_increase.append(tt[i]-tt[i-1])
    y.append(current_epi)
data['data']

X = []
for i in range(1, len(y)+1):
    X.append([i])

X

# vertical line corresponding to the beginning of restriction laws. 
di = 14
restrictions_x = [di,di,di,di,di,di]
restrictions_y = [0,10,20,30,40,50]

# vertical line corresponding to the beginning of effects of restriction laws (after a week)
de = di + 7
effects_x = [de,de,de,de,de,de]
effects_y = [0,10,20,30,40,50]
de

import matplotlib.pyplot as plt

# plt.scatter(X, y,  color='black')
# plt.plot(restrictions_x,restrictions_y, color='red', linewidth=2)
# plt.plot(effects_x,effects_y, color='green', linewidth=2)
# plt.grid()
# plt.xlabel('Days')
# plt.xlim(0,40)
# plt.ylim(0,50)
# plt.xticks([0,5,10,15,20,25,30,35,40],
#            ["24 Febr", "29 Febr", "5 Mar", "10 Mar", "15 Mar", "20 Mar", "25 Mar", "30 Mar", "4 Apr"])

# plt.ylabel('Epidemics Progression Index (EPI)')
# plt.savefig("EPI-all.png")
# plt.show()

import numpy as np
from sklearn import linear_model

# alleno il modello solo a partire dagli effetti del cambiamento
X = X[de:]
y = y[de:]

print(X)
# Linear Regression
linear_regr = linear_model.LinearRegression()

# Train the model using the training sets
linear_regr.fit(X, y)


linear_regr.score(X,y)

from sklearn.metrics import max_error
import math

y_pred = linear_regr.predict(X)
error = max_error(y, y_pred)
error

X_test = []

gp = 40

for i in range(de, de + gp):
    X_test.append([i])

y_pred_linear = linear_regr.predict(X_test)

y_pred_max = []
y_pred_min = []
for i in range(0, len(y_pred_linear)):
    y_pred_max.append(y_pred_linear[i] + error)
    y_pred_min.append(y_pred_linear[i] - error)

# calcolo la data iniziale degli effetti del cambiamento
from datetime import datetime
from datetime import timedelta  

data_eff = datetime.strptime(data['data'][de], '%Y-%m-%dT%H:%M:%S')
# date previsione
date_prev = []
x_ticks = []
step = 5
data_curr = data_eff
x_current = de
n = int(gp/step)
for i in range(0, n):
    date_prev.append(str(data_curr.day) + "/" + str(data_curr.month))
    x_ticks.append(x_current)
    data_curr = data_curr + timedelta(days=step)
    x_current = x_current + step

    plt.grid()
# plt.scatter(X, y,  color='black')

# plt.plot(X_test, y_pred_linear, color='green', linewidth=2)
# plt.plot(X_test, y_pred_max, color='red', linewidth=1, linestyle='dashed')
# plt.plot(X_test, y_pred_min, color='red', linewidth=1, linestyle='dashed')

# plt.xlabel('Days')
# plt.xlim(de,de+gp)

# plt.xticks(x_ticks, date_prev)
# plt.ylabel('Epidemics Progression Index (EPI)')
# plt.yscale("log")

# plt.savefig("EPI-prediction.png")
# plt.show()

def n_to_date(n):
    return data_eff + timedelta(days=n-de)
data_zero = round(- linear_regr.intercept_ / linear_regr.coef_[0])
n_to_date(data_zero)
def build_line(x1,y1,x2,y2):
    m = float(y2 - y1)/(x2-x1)
    q = y1 - (m*x1)
    return [m,q]
import math
line_max = build_line(X_test[0][0], y_pred_max[0], X_test[1][0], y_pred_max[1])
data_zero_max = math.ceil(- line_max[1] / line_max[0])
n_to_date(data_zero_max)
line_min = build_line(X_test[0][0], y_pred_min[0], X_test[1][0], y_pred_min[1])
data_zero_min = math.floor(- line_min[1] / line_min[0])
n_to_date(data_zero_min)
def date_to_n(my_date):
    initial_date = datetime.strptime(data['data'][0], '%Y-%m-%dT%H:%M:%S')
    return (my_date - initial_date).days + 1

my_date = datetime.strptime("2020-04-05", '%Y-%m-%d')
n = date_to_n(my_date)
predict = linear_regr.predict([[n]])
predict[0]
def average(mylist):
    return sum(mylist)/len(mylist)

# calculate the plateau considering the average increase of swabs
def plateau(y_pred,data_end,metrics):
    avg_tt = metrics(tt_increase[de:])

    np_avg = []
    #np_start = data['totale_casi'][len(data['totale_casi'])-1]
    np_start = data['totale_casi'][de]
    np_avg.append(np_start)

    for i in range(0, data_end-de):
        np = y_pred[i]*avg_tt/100 + np_avg[i-1]
        np_avg.append(np)
        
    last_value = max(np_avg)
    for i in range(0, gp-len(np_avg)):
        np_avg.append(last_value)
    return np_avg
plateau_min = plateau(y_pred_min,data_zero_min, max)
plateau_max = plateau(y_pred_max,data_zero_max, max)
plateau_avg = plateau(y_pred_linear,int(data_zero), max)
plt.plot(X_test,plateau_min, color='red', linewidth=1, linestyle='dashed')
plt.plot(X_test,plateau_max, color='red', linewidth=1, linestyle='dashed')
plt.plot(X_test,plateau_avg, color='green', linewidth=2)
plt.scatter(X,tc[de+1:], color='black', linewidth=2)
plt.xlabel('Days')
plt.xlim(de,de+gp)
#plt.ylim(0,50)
plt.xticks(x_ticks, date_prev)
#plt.yticks([0,20,30,40,50,60])

plt.ylabel('Total number of positives')
plt.grid()
plt.savefig("TNP.png")
plt.show()
max(plateau_min)
max(plateau_max)
max(plateau_avg)
