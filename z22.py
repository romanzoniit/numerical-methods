import numpy as np
import matplotlib.pyplot as plt


# define the general model

def TestEquation(t, y):

    #xdot = np.array([alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])
    ydot = np.array([-y[1] + t**2 + 6*t + 1,
                     y[0] - 3*t**2 + 3*t + 1])
    """xdot[0] = -x[1] + t**2 + 6*t + 1
    xdot[1] = x[0] - 3*t**2 + 3*t + 1
    xdot[2] = -x[2] + t**3 + 3
    xdot[3] = x[2]*x[1] + t**2"""
    return ydot


def LotkaVolterraModelN(tau, y, sigma):
    """a = params1["a"]
    b = params1["b"]
    c = params1["c"]
    d = params1["d"]"""
    ydot = np.zeros([2])
    ydot[0] = y[0] - y[0] * y[1]
    ydot[1] = -sigma * y[1] + y[0] * y[1]
    #ydot[0] = d/a * y[0] - d/a * y[0] * b/a * y[1]
    #ydot[1] = -sigma*y[1] + d/a * y[0] * b/a *y[1]
    #xdot[2] = d/a * x[0] * b/a * x[1]
    return ydot

def RungeKutta4(f, y0, dt, t):

    nt = t.size
    nx = y0.size
    y = np.zeros((nx, nt))
    y[:, 0] = y0

    for k in range(nt - 1):
        k1 = dt * f(t[k], y[:, k])

        print("вывод л1 =", k1)
        k2 = dt * f(t[k] + dt / 2, y[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, y[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, y[:, k] + k3)

        dy = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        y[:, k + 1] = y[:, k] + dy

    return y, t

def ExactSolution(t,y0):
    nt = t.size
    nx = y0.size
    y = np.zeros((nx, nt))
    y[:, 0] = y0
    y = np.array([3 * (t ** 2) - t - 1 + np.cos(t) + np.sin(t),
                  t ** 2 + 2 - np.cos(t) + np.sin(t)])
    return y

def TestGraph(t, y, y_exat):

    #y1 = lambda t1: 3 * (t1 ** 2) - t1 - 1 + np.cos(t1) + np.sin(t1)
    #y2 = lambda t1: t1 ** 2 + 2 - np.cos(t1) + np.sin(t1)
    plt.subplot(1, 2, 1)
    plt.title("Метод Рунге-Кутты")
    plt.plot(t, y[0, :], "r", label="y1")
    plt.plot(t, y[1, :], "b", label="y2")
    plt.xlabel("Time(t)")
    plt.ylabel("f")
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Точное решение")
    plt.plot(t, y_exat[0], "r", label="y1")
    plt.plot(t, y_exat[1], "b", label="y2")
    plt.xlabel("Time(t)")
    plt.ylabel("f")
    plt.grid()
    plt.legend()
    plt.show()



def ErrorDepend(t0, tf, y0, f):
    error1 = np.array([])
    error2 = np.array([])
    h_list = np.array([])

    for i in np.arange(1, 4):
        n = 10 + i
        h = (tf - t0) / n
        t = np.arange(t0, tf, h)
        h_list = np.append(h_list, h)
        y, t = RungeKutta4(f, y0, h, t)
        y_exat = ExactSolution(t, y0)
        print(y[0])
        d_temp1 = np.max(np.abs(y[0] - y_exat[0]))
        d_temp2 = np.max(np.abs(y[1] - y_exat[1]))
        error1 = np.append(error1, d_temp1)
        error2 = np.append(error2, d_temp2)
        print(n)
    #print(np.min(h_list), np.min(error1 / (h * h * h * h)))
    #print(np.min(h_list), np.min(error2 / (h * h * h * h)))
    plt.title("Зависимость максимальной погрешности e/h^4 от шага h ")
    print("maaaaaaaaaaaaaaaaaaaaaaaaaaaaaax")
    print(np.max(error1/ (h * h * h * h)))
    print(np.max(error2/ (h * h * h * h)))
    print("miiiiiiiiiiiiiiiiiiiiiiiin")
    print(np.min(error1 / (h * h * h * h)))
    print(np.min(error2 / (h * h * h * h)))
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("error")
    plt.xlabel("h")
    plt.plot(h_list, np.array(error1 / (h * h * h * h)), color='r', label="y1")
    plt.plot(h_list, np.array(error2 / (h * h * h * h)), color='b', label="y2")
    plt.grid()
    plt.legend()
    plt.show()

def ErrorDependh(t0, tf, y0, f):
    error1 = np.array([])
    error2 = np.array([])
    h_list = np.array([])

    for i in np.arange(1, 4):
        n = 10 + i
        h = (tf - t0) / n
        t = np.arange(t0, tf, h)
        h_list = np.append(h_list, h)
        y, t = RungeKutta4(f, y0, h, t)
        y_exat = ExactSolution(t, y0)
        d_temp1 = np.max(np.abs(y[0] - y_exat[0]))
        d_temp2 = np.max(np.abs(y[1] - y_exat[1]))
        error1 = np.append(error1, d_temp1)
        error2 = np.append(error2, d_temp2)
        print(n)
    print(np.min(h_list), np.min(error1 ))
    print(np.min(h_list), np.min(error2 ))
    plt.title("Зависимость максимальной погрешности e от шага h ")
    print(error1)
    print(error2)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel("error")
    plt.xlabel("h")
    plt.plot(h_list, np.array(error1), color='r', label="y1")
    plt.plot(h_list, np.array(error2), color='b', label="y2")
    plt.grid()
    plt.legend()
    plt.show()



def plotgraph(t, y, y_exat):
    plt.title("Абсолютная ошибка")
    plt.xlabel("t")
    plt.ylabel("Ошибка")
    plt.plot(t, np.abs(y[0, :] - y_exat[0, :]), color='r', label="y1")
    plt.plot(t, np.abs(y[1, :] - y_exat[1, :]), color='b',label="y2")
    # plt.plot(x, y1, color='red', linestyle="dotted", linewidth=3)
    plt.grid()
    plt.legend()
    plt.show()

def lotkaVolterraGraph():
    #params = {"a": 0.6, "b": 0.3, "c": 0.3, "d": 0.05}
    t0 = 0
    tf = 10
    dt = 0.1
    tau = np.arange(t0, tf, dt)
    all_sigma = np.arange(0.25, 4, 0.25)

    fig1, axs1 = plt.subplots(2, len(all_sigma))
    vecMax = []
    for idx, sigma in enumerate(all_sigma):
        y0 = np.array([2, 1])
        f = lambda tau, y: LotkaVolterraModelN(tau, y, sigma)
        y, t = RungeKutta4(f, y0, dt, tau)
        vecMax.append(y[1])
        axs1[0, idx].set_title(f'sigma={str(sigma)}',{'fontsize':'7'})
        fig1.suptitle("Графики решений (X(t),t) и (Y(t),t) при различных sigma")
        print(type(axs1[0,0]))
        axs1[0, idx].plot(tau, y[0, :], "r", label='X(t)')
        axs1[0, idx].plot(tau, y[1, :], "b", label='Y(t)')
        axs1[0, 0].set_xlabel("Time(t)")
        axs1[0, 0].set_ylabel("f(t)")

        # plt.plot(t,x[2, :], "g" , label="Predators")

        plt.grid()
        plt.legend()
        axs1[1, idx].plot(y[0, :], y[1, :])
        axs1[1,all_sigma.size//2].set_title("Фазовая диаграмма (X,Y)")
        #plt.ylim(top=3)  # adjust the top leaving bottom unchanged
        #plt.ylim(bottom=0)
        plt.xlabel("Preys-y1 ")
        plt.ylabel("Predators-y2")
        plt.grid()
    maxx = np.max(vecMax)
    axs1[1] = list(map(lambda x: x.set_ylim([0, maxx+0.1]), axs1[1]))
    plt.grid()
    plt.legend()
    plt.show()


def depend_sigma():
    params = {"a": 0.6, "b": 0.3, "c": 0.3, "d": 0.05}
    t0 = 0
    tf = 10
    dt = 0.1
    tau = np.arange(t0, tf, dt)
    y0 = np.array([2, 1])
    for sigma in np.arange(0.1, 4, 0.1):
        f = lambda tau, y: LotkaVolterraModelN(tau, y, sigma)
        y, t = RungeKutta4(f, y0, dt, tau)
        plt.plot(y[0, :], y[1, :])
    plt.title("Фазовая диаграмма (X,Y) при различных sigma")
    plt.xlabel("Preys-X ")
    plt.ylabel("Predators-Y")
    plt.grid()
    plt.legend()
    plt.show()



t0 = 0
tf = 3.2
dt = 0.2
t = np.arange(t0, tf, dt)
y0 = np.array([0, 1])
f_test = lambda t, y: TestEquation(t, y)
y, t = RungeKutta4(f_test, y0, dt, t)
y_exat = ExactSolution(t, y0)
from prettytable import PrettyTable
mytable = PrettyTable()
mytable.add_column("T", t)
mytable.add_column(" Y1 приближенное  ", y[0])
mytable.add_column(" Y1 точное ", y_exat[0])
mytable.add_column(" Y1 погрешность ", abs(y[0]-y_exat[0]))
mytable.add_column(" Y2 приближенное  ", y[1])
mytable.add_column(" Y2 точное ", y_exat[1])
mytable.add_column(" Y2 погрешность ", abs(y[1]-y_exat[1]))
print(mytable)

#TestGraph(t, y, y_exat)
#plotgraph(t, y, y_exat)
ErrorDependh(t0, tf, y0, f_test)
ErrorDepend(t0, tf, y0, f_test)
#lotkaVolterraGraph()
#depend_sigma()

print(2*10**(-4)*(2.5*10**-1)**(-4))
print(6*10**(-2)*(2.5*10**-2)**(-4))


"""params = {"alpha": 0.18, "beta": 0.03, "gamma": 0.45, "delta": 0.03}
params1 = {"a": 0.2, "b": 0.05, "c": 0.55, "d": 0.05}
x0 = np.array([20, 5])
t0 = 0
tf = 100
dt = 0.001
t = np.arange(t0, tf, dt) * params1["a"]

f = lambda t, x: LotkaVolterraModelN(x, params1, t)

print(f)

x, t = RungeKutta4(f, x0, t0, tf, dt, params1)

print(x)
print(t)
y1 = lambda t1: 3*(t1**2) - t1 - 1 + np.cos(t1) + np.sin(t1)
y2 = lambda t1: t1**2 + 2 - np.cos(t1) + np.sin(t1)
"""
"""#x1 = np.linspace(0, 3, 300)
plt.subplot(1, 2, 1)
plt.plot(t, x[0, :], "r", label="y1")
plt.plot(t, x[1, :], "b", label="y2")
#plt.plot(t, x[2, :], "g", label="y3")
#plt.plot(t, x[3, :], "black", label="y4")
plt.xlabel("Time(t)")
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x1, y1(x1), label="y1")
plt.plot(x1, y2(x1), label="y2")
plt.xlabel("Time(t)")
plt.grid()
plt.legend()
plt.show()


ax = plt.axes(projection='3d')

# Data for a three-dimensional line

ax.plot3D(x[0, :], x[1, :], 'gray')
#plt.subplot(1, 2, 2)
""""""plt.plot(x[0, :], x[1, :],x[2, :])
plt.xlabel("Preys")
plt.ylabel("Predators")
plt.grid()"""
"""
plt.show()


# импортируем модули
import numpy as np
import matplotlib.pyplot as plt
# функция
y1 = lambda t: 3*t**2 - t - +np.cos(t) + np.sin(t)
y2 = lambda t: t**2 +2 - np.cos(t) + np.sin(t)
# создаём рисунок с координатную плоскость
fig = plt.subplots()
# создаём область, в которой будет
# - отображаться график
x = np.linspace(0, 3,100)
# значения x, которые будут отображены
# количество элементов в созданном массиве
# - качество прорисовки графика
# рисуем график
plt.plot(x, y1(x), label="y1")
plt.plot(x, y2(x), label="y2")

# показываем график
plt.show()"""

