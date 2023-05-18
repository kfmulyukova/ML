import numpy as np
import itertools
import math
import plotly.graph_objects as go


def f_x(x):
    return x
def f_x2(x):
    return x*x
def f_x3(x):
    return np.power(x, 3)

def numb_of_funcs(func):
    n_of_funcs = 0
    if isinstance(func, np.ufunc) or func == f_x or func == f_x2 or func == f_x3:
        n_of_funcs = 1
    else:
        for f in func:
            n_of_funcs += 1
    return n_of_funcs

def design_matrix(N, func, x):
    if numb_of_funcs(func) == 1:
        design_matrix = np.array([np.ones(N), func(x)]).T
    elif numb_of_funcs(func) == 2:
        design_matrix = np.array([np.ones(N), func[0](x), func[1](x)]).T
    else:
        design_matrix = np.array([np.ones(N), func[0](x), func[1](x), func[2](x)]).T
    return np.nan_to_num(design_matrix, nan=0.000000001)

def find_w(N, func, x, t_n):
    phi = design_matrix(N, func, x)
    #print(phi)
    w = np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), t_n)
    return w

def regression(func, x, w):
    y_list = []
    for i in x:
        if numb_of_funcs(func) == 1:
            y = w[0] + np.dot(w[1], np.nan_to_num(func(x), nan=0.000000001))
        elif numb_of_funcs(func) == 2:
            y = w[0]+np.dot(w[1], np.nan_to_num(func[0](x),
                                            nan=0.000000001))+np.dot(w[2], np.nan_to_num(func[1](x), nan=0.000000001))
        else:
            y = w[0] + np.dot(w[1], np.nan_to_num(func[0](x),
                                            nan=0.000000001)) + np.dot(w[2], np.nan_to_num(func[1](x),
                                            nan=0.000000001)) + np.dot(w[3], np.nan_to_num(func[2](x),
                                            nan=0.000000001))
        y_list.append(np.nan_to_num(y))
    return y_list

def RMSE(t_n, regr_list):
    return np.sqrt(np.mean(np.power((t_n - regr_list), 2)))

x = np.linspace(0, 2*np.pi, 1000)

avg, disp, n = 0, 100, 1000

np.random.shuffle(x)
train_x = x[:800]
valid_x = x[800:900]
test_x = x[900:]

train = np.array([train_x, np.random.normal(loc=avg, scale=np.sqrt(disp), size=len(train_x))])
valid = np.array([valid_x, np.random.normal(loc=avg, scale=np.sqrt(disp), size=len(valid_x))])
test = np.array([test_x, np.random.normal(loc=avg, scale=np.sqrt(disp), size=len(test_x))])

gr_truth_train = []
gr_truth_valid = []
gr_truth_test = []

for i in train_x:
    gr_truth_train.append(100*np.sin(i)+0.5*math.exp(i)+300)
for i in valid_x:
    gr_truth_valid.append(100*np.sin(i)+0.5*math.exp(i)+300)
for i in test_x:
    gr_truth_test.append(100*np.sin(i)+0.5*math.exp(i)+300)

t_n_train = gr_truth_train + train[1]
t_n_valid = gr_truth_valid + valid[1]
t_n_test = gr_truth_test + test[1]

funcs_list = [np.sin, np.cos, np.log, np.exp, np.sqrt, f_x, f_x2, f_x3]

regr_list1, regr_list2, regr_list3 = [], [], []
w_and_func1, w_and_func2, w_and_func3 = [], [], []

for current_func in funcs_list:
    w_and_func1.append([find_w(800, current_func, train[0], t_n_train),
                       current_func])
for current_func in itertools.combinations(funcs_list, 2):
    w_and_func2.append([find_w(800, current_func, train[0], t_n_train),
                       current_func])
for current_func in itertools.combinations(funcs_list, 3):
    w_and_func3.append([find_w(800, current_func, train[0], t_n_train),
                       current_func])

w_and_func = w_and_func1 + w_and_func2 + w_and_func3
rmse_list = []


for i in range(len(w_and_func1)):
    rmse_list.append(RMSE(t_n_valid, regression(w_and_func1[i][1], valid_x, w_and_func1[i][0])))
for i in range(len(w_and_func2)):
    rmse_list.append(RMSE(t_n_valid, regression(w_and_func2[i][1], valid_x, w_and_func2[i][0])))
for i in range(len(w_and_func3)):
    rmse_list.append(RMSE(t_n_valid, regression(w_and_func3[i][1], valid_x, w_and_func3[i][0])))


rmse_list_num = list(enumerate(rmse_list, 0))
print(rmse_list_num)
rmse_list_num = sorted(rmse_list_num, key=lambda x: x[::-1])
print(rmse_list_num)
index1, index2, index3 = rmse_list_num[0][0], rmse_list_num[1][0], rmse_list_num[2][0]


valid_conf = np.array([rmse_list_num[0][1], rmse_list_num[1][1], rmse_list_num[2][1]])
train_regr1 = regression(w_and_func[index1][1], train_x, w_and_func[index1][0])
train_regr2 = regression(w_and_func[index2][1], train_x, w_and_func[index2][0])
train_regr3 = regression(w_and_func[index3][1], train_x, w_and_func[index3][0])

train_conf = np.array([RMSE(t_n_train, train_regr1), RMSE(t_n_train, train_regr2), RMSE(t_n_train, train_regr3)])
weights1, weights2, weights3 = w_and_func[index1][0], w_and_func[index2][0], w_and_func[index3][0]

funcs_list = [np.sin, np.cos, np.log, np.exp, np.sqrt, f_x, f_x2, f_x3]
def f_name(f_name):
    if f_name == np.sin:
        f_name = 'sin(x)'
    elif f_name == np.cos:
        f_name = 'cos(x)'
    elif f_name == np.log:
        f_name = 'ln(x)'
    elif f_name == np.exp:
        f_name = 'e^x'
    elif f_name == np.sqrt:
        f_name = 'sqrt(x)'
    elif f_name == f_x:
        f_name = 'x'
    elif f_name == f_x2:
        f_name = 'x^2'
    elif f_name == f_x3:
        f_name = 'x^3'
    return f_name

func_names1 = w_and_func[index1][1]
func_names2 = w_and_func[index2][1]
func_names3 = w_and_func[index3][1]

test_rmse = RMSE(t_n_test, regression(func_names1, test_x, weights1))

'''if isinstance(func_names1, np.ufunc) or func_names1 == f_x or func_names1 == f_x2 or func_names1 == f_x3:
    func_names1 = f_name(func_names1)
if isinstance(func_names2, np.ufunc) or func_names2 == f_x or func_names2 == f_x2 or func_names2 == f_x3:
    func_names2 = f_name(func_names2)
if isinstance(func_names3, np.ufunc) or func_names3 == f_x or func_names3 == f_x2 or func_names3 == f_x3:
    func_names3 = f_name(func_names3)'''


reg1 = "".join([f"{w:.2f}*{f_name(name)}+" for w, name in zip(weights1[1:], func_names1)])+\
       f"{weights1[0]:.2f}<br>Ошибка на тестовой выборке:{test_rmse}"
reg2 = "".join([f"{w:.2f}*{f_name(name)}+" for w, name in zip(weights2[1:], func_names2)])+f"{weights2[0]:.2f}"
reg3 = "".join([f"{w:.2f}*{f_name(name)}+" for w, name in zip(weights3[1:], func_names3)])+f"{weights3[0]:.2f}"
labels = [reg1, reg2, reg3]
fig = go.Figure(data=[go.Bar(x=labels, y=train_conf, name="train confidence"),
                      go.Bar(x=labels, y=valid_conf, name="valid confidence")])
fig.write_html("bars.html")
fig.show()
