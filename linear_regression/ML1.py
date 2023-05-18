import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def f_rand(avg, disp, n):
    s = np.sqrt(disp)
    values = np.random.normal(loc=avg, scale=s, size=n)
    return values

def phi_d(x_i, degree):
    phi_d = []
    phi_d_sum = 0
    for i in range(0, degree+1):
        phi_d.append(np.power(x_i, i))
    return (phi_d)

def find_w(N, degree, t_n):
    phi = np.zeros((N, degree+1))
    phi_list = []
    for i in range(0, N):
        for j in range(0, degree+1):
            phi_list.append(np.power(x[i], j))
        phi[i] = phi_list
        phi_list.clear()
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), t_n)
    return w

def regression(degree, N, t_n, x):
    w = find_w(N, degree, t_n)
    j = 1
    y = []
    y_list = []
    for i in x:
        y = np.dot(w[1:], phi_d(i, degree)[1:])
        y_sum = w[0] + y
        y_list.append(y_sum)
    return y_list

def RMSE(N, degree, t_n, x, regr_list):
    rmse_list = []
    rmse_list = []
    for i in regr_list:
        almost_rmse = np.sqrt(np.mean((t_n - i) ** 2))
        rmse_list.append(almost_rmse)
    rmse = np.sqrt(np.mean(rmse_list))
    return rmse


x = np.linspace(0, 2*np.pi, 1000)
ground_truth = []

for i in x:
    gr_truth = 100*np.sin(i)+0.5*math.exp(i)+300
    ground_truth.append(gr_truth)

avg, disp, n = 0, 100, 1000
values = f_rand(avg, disp, n)

t_n = ground_truth + values


fig = make_subplots(
rows=5,
cols=4,
start_cell="top-left",
subplot_titles=["Полином 1 степени", "Полином 2 степени", "Полином 3 степени", "Полином 4 степени",
                "Полином 5 степени", "Полином 6 степени", "Полином 7 степени", "Полином 8 степени",
                "Полином 9 степени", "Полином 10 степени", "Полином 11 степени", "Полином 12 степени",
                "Полином 13 степени", "Полином 14 степени", "Полином 15 степени", "Полином 16 степени",
                "Полином 17 степени", "Полином 18 степени", "Полином 19 степени", "Полином 20 степени"
                 ],
x_title="x",
y_title="y")

d, i, j, cols, rows = 1, 1, 1, 4, 5
degrees = 20
N = 1000
regr_list = []
while i <= cols and j <= rows and d <= degrees:
    fig.add_trace(go.Scatter(x=x, y=t_n, name='исходные данные', mode='markers',
                             marker=dict(color='LightSkyBlue', size=3)), row=j, col=i)
    fig.add_trace(go.Scatter(x=x, y=ground_truth, name="gt"), row=j, col=i)
    regr = regression(d, N, t_n, x)
    regr_list.append(regr)
    fig.add_trace(go.Scatter(x=x, y=regr, name="regression"), row=j, col=i)
    i += 1
    d += 1
    if i == cols + 1:
        i = 1
        j += 1
fig.show()
fig.write_html("graph.html")


fig2 = make_subplots(
start_cell="top-left",
subplot_titles=["RMSE"],
x_title="степень",
y_title="значение RMSE")
y_rmse = []
for i in range(1, 21):
    y_rmse.append(RMSE(1000, 20, t_n, x, regr_list[i-1]))
fig2.add_trace(go.Scatter(x=np.arange(1, 21), y=y_rmse, name="RMSE"))
fig2.show()
fig2.write_html("RMSE.html")
go.Scatter()
