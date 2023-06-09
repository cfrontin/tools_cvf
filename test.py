import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tools_cvf

Nplot = 201
x = np.linspace(0, 2 * np.pi, Nplot)


def do(fname_save=None):
    fig, ax = plt.subplots()
    for idx in range(5):
        y = np.cos(x - idx * np.pi / 8)
        plt.plot(x, y)
    plt.xlabel("$x$ variable")
    plt.ylabel("$y$ variable")
    if fname_save is not None:
        plt.savefig(fname_save)
    plt.show()


# do()

url_stylesheet_cvf = "https://github.nrel.gov/raw/cfrontin/tools_cvf/main/tools_cvf/stylesheet_cvf.mplstyle"
url_stylesheet_cvf_notex = "https://github.nrel.gov/raw/cfrontin/tools_cvf/main/tools_cvf/stylesheet_cvf_notex.mplstyle"
url_stylesheet_seaborn = "https://github.nrel.gov/raw/cfrontin/tools_cvf/main/tools_cvf/stylesheet_seaborn.mplstyle"
url_stylesheet_nrel = "tools_cvf/stylesheet_nrel.mplstyle"

use_tex = True
if use_tex:
    if True:
        # with plt.style.context(tools_cvf.get_stylesheets()):
        # with plt.style.context(['/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_seaborn.mplstyle',
        #                         '/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_cvf.mplstyle']):
        with plt.style.context([url_stylesheet_seaborn, url_stylesheet_cvf]):
            do(fname_save="assets/plot_stylesheet.png")
    if True:
        # with plt.style.context(tools_cvf.get_stylesheets()):
        # with plt.style.context(['/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_seaborn.mplstyle',
        #                         '/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_cvf.mplstyle']):
        with plt.style.context(
            [url_stylesheet_seaborn, url_stylesheet_cvf, url_stylesheet_nrel]
        ):
            do(fname_save="assets/plot_stylesheet.png")
    if True:
        # with plt.style.context(tools_cvf.get_stylesheets(dark= True)):
        # with plt.style.context(['dark_background',
        #                         '/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_seaborn.mplstyle',
        #                         '/Users/cfrontin/codes/tools_cvf/tools_cvf/stylesheet_cvf.mplstyle',]):
        with plt.style.context(
            [
                "dark_background",
                url_stylesheet_seaborn,
                url_stylesheet_cvf,
            ]
        ):
            do(fname_save="assets/plot_stylesheet_dark.png")
else:
    if True:
        with plt.style.context(tools_cvf.get_stylesheets(use_latex=False)):
            # with plt.style.context([url_stylesheet_seaborn, url_stylesheet_cvf_notex]):
            do(fname_save="assets/plot_stylesheet.png")
    if True:
        with plt.style.context(tools_cvf.get_stylesheets(dark=True, use_latex=False)):
            do(fname_save="assets/plot_stylesheet_dark.png")
