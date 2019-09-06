from adaptive_fit import *


def deep_reg(x, y, figure=0, epochs=20000, adaptive=False, plot=False):

    if adaptive:
        regression = fit_adaptive(x, y, epochs=epochs)
    else:
        regression = fit(x, y, epochs=epochs)

    if plot:
        plot_regression(regression, color='r', label='w', figure=figure,
                        linspace=[x.min(), x.max()], x=x, y=y, Percent=1)

    return regression