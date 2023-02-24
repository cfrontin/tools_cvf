
# Cory's plotting library

This repo exists for two reasons:
  1. centralize my common plotting stylesheets and options for uniformity
  2. make scripts to quickly generate common plots, but nice looking

## Stylesheet usage

`plot_cvf` provides a function `get_stylesheets` which gets the stylesheets I
like to use from where they're installed for the system, and returns a list of
sheets to apply based on the provided options, for integration into matplotlib:

```
import matplotlib.pyplot as plt
import plot_cvf

plt.style.use(plot_cvf.get_stylesheets())

... do standard plotting with pyplot ...

```

will result in:

![dressed up pyplot sinusoid](assets/plot_stylesheet.png)
![dressed up pyplot sinusoid (dark)](assets/plot_stylesheet_dark.png)



