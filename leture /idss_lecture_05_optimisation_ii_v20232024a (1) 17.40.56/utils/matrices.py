try:
    import sympy, IPython.display
    sympy.init_printing(use_latex='png')
except: 
    sympy = False
    
def print_matrix(name, matrix):
    if len(matrix.shape)==1:
        matrix = matrix[None,:]
        
    if sympy:    
        IPython.display.display(IPython.display.Latex("${0} = {1}$".format(name, sympy.latex(sympy.Matrix(matrix)))))
    else:
        print(name, "\n", matrix)

from matplotlib.patches import Polygon

import numpy as np
import matplotlib.pyplot as plt

def show_matrix_effect(m, suptitle=""):

    def piped(ax, pts):
        f = 1.08
        a, b, c, d = (pts[:, 0] * f, pts[:, 9] * f, pts[:, 90] * f,
                      pts[:, 99] * f)

        poly = Polygon(
            [a, b, d, c], facecolor=[0.5, 0.75, 0.75, 0.2], edgecolor='k')
        ax.add_patch(poly)
        ax.text(
            a[0], a[1], 'A', color='r', fontsize=30, ha='center', va='center')
        ax.text(b[0], b[1], 'B', fontsize=20, ha='center', va='center')
        ax.text(c[0], c[1], 'C', fontsize=20, ha='center', va='center')
        ax.text(d[0], d[1], 'D', fontsize=20, ha='center', va='center')

    fig = plt.figure()
    lax = fig.add_subplot(1, 2, 1)
    rax = fig.add_subplot(1, 2, 2)
    
    
    # 2D transform
    box_x, box_y = np.meshgrid(
        np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    box = np.stack([box_x.reshape([-1]), box_y.reshape([-1])])
    color = np.linspace(0, 1, box.shape[1])
    # do the transform -- just one line
    if m.shape[0]==3:
        mbox = np.concatenate([box, np.ones((1,box.shape[1]))], axis=0)        
        transformed = np.dot(m, mbox)[:-1,:]        
        
    else:
        transformed = np.dot(m, box)
    lax.scatter(box[0, :], box[1, :], c=color, cmap='viridis')
    rax.scatter(
        transformed[0, :], transformed[1, :], c=color, cmap='viridis')
    piped(lax, box)
    piped(rax, transformed)

    # decorate the axes
    lax.set_title("Original")
    rax.set_title("Transformed")
    lax.scatter([0], [0], marker='o', color='k')
    rax.scatter([0], [0], marker='o', color='k')
    lax.axvline(0, color='k', ls=':')
    rax.axvline(0, color='k', ls=':')
    lax.axhline(0, color='k', ls=':')
    rax.axhline(0, color='k', ls=':')

    lax.axis("equal")
    lax.axis("off")
    rax.axis("equal")
    rax.axis("off")

    lax.set_xlim(-1.5, 1.5)
    rax.set_xlim(-1.5, 1.5)
    lax.set_ylim(-1.5, 1.5)
    rax.set_ylim(-1.5, 1.5)        
    plt.suptitle(suptitle)