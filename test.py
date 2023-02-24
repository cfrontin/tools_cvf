
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import plot_cvf

Nplot= 201
x= np.linspace(0, 2*np.pi, Nplot)

def do(fname_save= None):
  fig, ax= plt.subplots()
  for idx in range(5):
    y= np.cos(x - idx*np.pi/8)
    plt.plot(x, y)
  plt.xlabel("$x$ variable")
  plt.ylabel("$y$ variable")
  if fname_save is not None:
    plt.savefig(fname_save)
  plt.show()

# do()

if True:
  # with plt.style.context(plot_cvf.get_stylesheets()):
  with plt.style.context(['/Users/cfrontin/codes/plot_cvf/plot_cvf/stylesheet_seaborn.mplstyle',
                          '/Users/cfrontin/codes/plot_cvf/plot_cvf/stylesheet_cvf.mplstyle']):
    do(fname_save= "assets/plot_stylesheet.png")
if True:
  # with plt.style.context(plot_cvf.get_stylesheets(dark= True)):
  with plt.style.context(['dark_background',
                          '/Users/cfrontin/codes/plot_cvf/plot_cvf/stylesheet_seaborn.mplstyle',
                          '/Users/cfrontin/codes/plot_cvf/plot_cvf/stylesheet_cvf.mplstyle',]):
    do(fname_save= "assets/plot_stylesheet_dark.png")
