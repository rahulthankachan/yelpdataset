from itertools import cycle
from matplotlib import pyplot as plt
import itertools
import numpy


def line_graph(y, x, line_labels, title, x_axis_name, y_axix_name):
    figure_num = len(list(map(plt.figure, plt.get_fignums()))) + 1
    figure1 = plt.figure(figure_num)

    cm = plt.get_cmap('gist_rainbow')
    ax = figure1.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / 10) for i in range(10)])

    for y_val, line_label in zip(y, line_labels):
        plt.plot(x, y_val, marker="o",label=line_label, linewidth=2)

    axes = plt.gca()
    axes.set_xlim([-5.2, 1.2])
    axes.set_ylim([0.3,0.6])
    plt.ylabel(y_axix_name)
    plt.xlabel(x_axis_name)
    plt.legend()
    plt.xticks(x)
    figure1.suptitle(title, fontsize=20)


def main():
    line_graph([
        [0.391497278031, 0.391855124355, 0.386610647627, 0.397703128407, 0.385597266586, 0.360151630975, 0.453617470063],
        [0.380670909969, 0.378436793487, 0.401407410428, 0.397606516031, 0.389104336664, 0.40882230614, 0.460008329277],
        [0.401629250267,0.400967933991, 0.383860097864, 0.398182888624,0.401622156413, 0.406284861329, 0.476693648206],
        [0.467206789865, 0.449818451336,0.451966326443,0.450320447585, 0.448155154708, 0.461859597376, 0.473458058214]
                ], [-5, -4, -3,-2,-1,0,1],
               ["1k data","5k data", "10k data","50k data","100k data"], "MLP lbfgs solver F1 scores variation with alpha values", 'alpha values(10e power)', 'F1 score')

    plt.show()



main()