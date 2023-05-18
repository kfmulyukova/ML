import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

f_height = np.random.randn(500) * 20 + 160
b_height = np.random.randn(500) * 10 + 190
height_of_all = np.concatenate((f_height, b_height))

def rand_classifier(input_height):
    return np.random.choice([0, 1], size=input_height.size)

def height_classifier(height, input_height):
    h_classifier = []
    for h in input_height:
        if h <= height:
            h_classifier.append(0)
        else:
            h_classifier.append(1)
    return h_classifier

def h_classifier_metrics(height):
    TP, TN, FP, FN = 0, 0, 0, 0
    for h in height_classifier(height, f_height):
        if h == 0:
            TN += 1
        else:
            FP += 1
    for h in height_classifier(height, b_height):
        if h == 1:
            TP += 1
        else:
            FN += 1
    accuracy = (TP + TN) / 1000
    if TP == 0:
        precision = 0
        recall = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    return np.array([TP, TN, FP, FN, accuracy, precision, recall])

def trapz(y, x):
    pair = tuple(zip(x, y))
    pair = sorted(pair, key=lambda x: x[0])
    S = 0
    for i in range(len(pair)-1):
        h = pair[i+1][0]-pair[i][0]
        S += h*(pair[i][1]+pair[i+1][1])/2
    return S

TP1, TN1, FP1, FN1 = 0, 0, 0, 0
for h in rand_classifier(f_height):
    if h == 0:
        TN1 += 1
    else:
        FP1 += 1
for h in rand_classifier(b_height):
    if h == 1:
        TP1 += 1
    else:
        FN1 += 1
accuracy1 = (TP1+TN1)/1000
precision1 = TP1/(TP1+FP1)
recall1 = TP1/(TP1+FN1)

print('Случайный классификатор:\n', 'TP:', TP1, 'TN:', TN1, 'FP:', FP1, 'FN:', FN1, 'accuracy:', accuracy1,
      'precision:', precision1, 'recall:', recall1)
metrics2 = h_classifier_metrics(175)
print('Ростовой классификатор:\n', 'TP:', metrics2[0], 'TN:', metrics2[1], 'FP:', metrics2[2], 'FN:', metrics2[3],
      'accuracy:', metrics2[4], 'precision:', metrics2[5], 'recall:', metrics2[6])

x_recall = []
y_precision = []
for i in np.arange(90, 231, 10):
    x_recall.append(h_classifier_metrics(i)[6])
    y_precision.append(h_classifier_metrics(i)[5])

fig2 = make_subplots(
start_cell="top-left",
x_title=f"Recall<br>Площадь по методу трапеций:{trapz(y_precision, x_recall)}",
y_title="Precision")

fig2.add_trace(go.Scatter(x=x_recall, y=y_precision))
fig2.show()
fig2.write_html("classifier.html")
go.Scatter()

print(trapz(y_precision, x_recall))
