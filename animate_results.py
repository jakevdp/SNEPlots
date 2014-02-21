import glob
import re

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib import animation

from sklearn.mixture import GMM


from compute_likelihoods import QuadraticModel, wCDMModel, wCDMModel2D


class AnimateResults(object):
    def __init__(self, z, mag, dmag, Model, interval=200):
        filepattern = 'traces/trace_{0}_*.npy'.format(Model.__name__)

        self.z = z
        self.mag = mag
        self.dmag = dmag
        self.interval = interval

        trace_files = sorted(glob.glob(filepattern))
        N = map(int, [f.split('.')[0].split('_')[-1] for f in trace_files])

        self.N, self.trace_files = zip(*sorted(zip(N, trace_files)))
        self.zfit = np.logspace(-1, 0, 10)
        
        self.points_alpha = 0.1 + np.zeros(len(self.N))
        self.lines_alpha = 0.2 + np.zeros(len(self.N))
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xscale('log')
        self.ax2 = self.fig.add_axes([0.65, 0.15, 0.2, 0.2])
        self.ax2.xaxis.set_major_locator(plt.MultipleLocator(0.3))
        self.ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))

        self.model = Model(self.z, self.mag, self.dmag)

        self.traces = []
        self.samples = []
        self.mag_fits = []

        for trace_file in self.trace_files:
            print trace_file
            trace = np.load(trace_file)
            trace = trace[:, 50:, :]

            sample = trace[:, -3:, :].reshape(-1, trace.shape[-1])
            trace = trace.reshape(-1, trace.shape[-1])

            print trace.mean(0)
            print trace.std(0)

            #gmm = GMM(1, 'full')
            #gmm.fit(trace)
            #sample = gmm.sample(50, random_state=0)

            mag_fit = self.model.eval(sample, self.zfit)

            self.traces.append(trace)
            self.samples.append(sample)
            self.mag_fits.append(mag_fit)
            
        
    def anim_init(self):
        # Here we set up the plot elements that we'll work with
        self.points, = self.ax.plot([1], [1], 'ok', ms=7, alpha=0.1)
        self.lines = self.ax.plot([1], np.ones((1, 50)), '-b', alpha=0.1)
        self.traceplot, = self.ax2.plot([1], [1], '.k', alpha=0.1)

        self.ax.set_xlim(0.1, 1)
        self.ax.set_ylim(19, 26)

        self.ax2.set_xlim(np.min([t[:, -2].min() for t in self.traces]),
                          np.max([t[:, -2].max() for t in self.traces]))
        self.ax2.set_ylim(np.min([t[:, -1].min() for t in self.traces]),
                          np.max([t[:, -1].max() for t in self.traces]))
                         

        self.ax.set_xlabel('redshift')
        self.ax.set_ylabel('magnitude')
    
        self.text = self.ax.text(0.01, 0.99, "",
                                 size=18, ha='left', va='top',
                                 transform=self.ax.transAxes)
    
        for line in self.lines:
            line.set_data([], [])
        self.points.set_data([], [])
        self.traceplot.set_data([], [])

        return self.lines + [self.traceplot, self.points, self.text]
    
    def anim_frame(self, i):
        Ni = self.N[i]
        zi = self.z[:Ni]
        magi = self.mag[:Ni]
        trace = self.traces[i]
        mag_fit = self.mag_fits[i]

        for line, mfit in zip(self.lines, mag_fit):
            line.set_data(self.zfit, mfit)
            line.set_alpha(self.lines_alpha[i])
        self.points.set_data(zi, magi)
        self.points.set_alpha(self.points_alpha[i])
        self.text.set_text("N = {0}".format(Ni))
        self.traceplot.set_data(self.traces[i][:, -2], self.traces[i][:, -1])
        
        return self.lines + [self.traceplot, self.points, self.text]
        
    def show(self):
        return animation.FuncAnimation(self.fig, self.anim_frame,
                                       init_func=self.anim_init,
                                       frames=len(self.N),
                                       interval=self.interval)

if __name__ == '__main__':
    z, mag = np.loadtxt('main.dat', unpack=True)
    dmag = 0.5

    anim = AnimateResults(z, mag, dmag, wCDMModel)
    A = anim.show()
    A.save('{0}.mp4'.format(anim.model.__class__.__name__), fps=5)
    plt.show()
