"""
Code to construct a supersample of supernova data
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.mixture import GMM


def construct_supersample(infile='main.dat', outfile='supersample.dat',
                          Nout=20000):
    data = np.loadtxt(infile)
    data[:, 0] = np.log(data[:, 0])

    pca = PCA(2).fit(data)
    data_pca = pca.transform(data)

    #model = KernelDensity(0.05)
    model = GMM(20, 'full', random_state=0)
    model.fit(data_pca)
    data2_pca = model.sample(Nout)
    data2 = pca.inverse_transform(data2_pca)
    data2[:, 0] = np.exp(data2[:, 0])
    np.savetxt(outfile, data2)


def plot_data(filename='main.dat'):
    z, mag = np.loadtxt(filename, unpack=True)
    fig, ax = plt.subplots()
    ax.plot(np.log(z), mag, '.k', alpha=0.05)
    ax.set_xlim(-3.5, 0.5)
    ax.set_ylim(16, 28)

construct_supersample(outfile='supersample.dat', Nout=1E6)
plot_data('main.dat')
plot_data('supersample.dat')
plt.show()
