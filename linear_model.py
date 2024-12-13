import numpy as np
import matplotlib.pyplot as plt

'''
Так как эмпирический риск в нашем случае - это кусочно непрерывная функция, то методом МНК мы минимизировать функцию не сможем, тогда есть друго способ:

1) Сначала инициализируем веса как вектор [w1, w2] = [0, -1]
2) Затем, если отступ М будет меньше нуля (то есть неверная классификация), то немного увеличиваем w1 в соответствии с формулой: 
w1 = w1+ n*yi
3) Затем вычисляем критерий качества, если он близок к нулю, останавливаем программу (решение найдено)
'''


x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(x_train) #длина выборки
w = [0, -1] #вектор весов
a = lambda x: np.sign(x[0]*w[0] + x[1]*w[1]) #знаковая функция модели
N = 50 #колличество итераций
L = 0.1 #шаг обучения
e = 0.1 #небольшая добавка, чтобы сдвинуть разделяющую линию

last_error_index = -1

for n in range(N):
    for i in range(n_train):
        margine = y_train[i]*a(x_train[i]) #отступ
        if margine < 0:
            w[0] += L * y_train[i]
            last_error_index = i
            
    Q = sum([1 for i in range(n_train) if y_train[i]*a(x_train[i])< 0]) #эперический риск
    if Q == 0:
        print("Обучение завершено")
        break
        
if last_error_index > -1:
    w[0] += e * y_train[last_error_index]

print(w)

line_x = list(range(max(x_train[:, 0])))
line_y = [w[0] * x for x in line_x]

x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("len")
plt.xlabel("weight")
plt.grid(True)
plt.show()
