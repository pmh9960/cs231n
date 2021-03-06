import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def show_dist_2d():
    x = y = np.linspace(-1, 1, 101)
    l1 = np.array([np.abs(i) + np.abs(j) for j in y for i in x])
    l2 = np.array([np.sqrt(i * i + j * j) for j in y for i in x])
    L1 = l1.reshape(101, 101)
    L2 = l2.reshape(101, 101)
    plt.subplot(1, 2, 1)
    plt.imshow(L1)
    plt.title("L1")
    plt.subplot(1, 2, 2)
    plt.imshow(L2)
    plt.title("L2")
    plt.show()


def show_dist_3d():
    x = y = np.linspace(-1, 1, 101)
    l1 = np.array([np.abs(i) + np.abs(j) for j in y for i in x])
    l2 = np.array([np.sqrt(i * i + j * j) for j in y for i in x])
    L1 = l1.reshape(101, 101)
    L2 = l2.reshape(101, 101)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("L1")
    surf = ax.plot_surface(x, y, L1, linewidth=0, cmap=cm.coolwarm, antialiased=False)
    ax.set_zlim(0, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("L2")
    surf = ax.plot_surface(x, y, L2, linewidth=0, cmap=cm.coolwarm, antialiased=False)
    ax.set_zlim(0, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    plt.show()


if __name__ == "__main__":
    show_dist_3d()
