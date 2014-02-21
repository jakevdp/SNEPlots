import numpy as np
import matplotlib.pyplot as plt


def compute_mag_fit(z, mag, z_fit):
    # simple stand-in for computing model from data
    p = np.polyfit(np.log(z), mag, 2)
    return np.polyval(p, np.log(z_fit))


def plot_first_N(N=None):
    z, mag = np.loadtxt('sne.txt').T

    # come up with some transparency gradient based on number of points
    alpha = min(1, 0.01 * 10000. / N)
    size = 5

    if N is None:
        N = len(z)

    z_fit = np.logspace(-1, 0, 100)
    mag_fit = compute_mag_fit(z, mag, z_fit)

    fig, ax = plt.subplots(subplot_kw={'xscale':'log'})
    ax.plot(z[:N], mag[:N], 'ok', ms=size, alpha=alpha)
    ax.plot(z_fit, mag_fit, '-k')

    ax.set_xlim(0.1, 1)
    ax.set_ylim(19, 26)

    ax.set_xlabel('redshift')
    ax.set_ylabel('magnitude')

    return fig, ax


if __name__ == '__main__':
    for N in [1000, 10000]:
        plot_first_N(N)
    
    plt.show()
