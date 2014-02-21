import os

import numpy as np
from scipy import optimize, interpolate
import emcee
from astropy import cosmology

import matplotlib.pyplot as plt
import time


class BaseModel(object):
    def __init__(self, z, mag, dmag):
        b = np.broadcast(z, mag, dmag)
        self.z = np.asarray(z, dtype=float)
        self.mag = np.asarray(mag, dtype=float)
        self.dmag = np.asarray(dmag, dtype=float)

    def _eval_single(self, theta, z):
        raise NotImplemetedError()

    def eval(self, theta, z):
        theta = np.asarray(theta)
        thetas = np.atleast_2d(theta).reshape(-1, theta.shape[-1])
        mag = np.array([self._eval_single(t, z) for t in thetas])
        return mag.reshape(theta.shape[:-1] + z.shape)

    def lnprior(self, theta):
        return 0

    def lnlike(self, theta):
        model_mag = self._eval_single(theta, self.z)
        return -0.5 * np.linalg.norm((self.mag - model_mag) / self.dmag)

    def lnprob(self, theta):
        prior = self.lnprior(theta)
        if np.isneginf(prior):
            return prior
        else:
            return prior + self.lnlike(theta)

    def chi2(self, theta):
        return -2 * self.lnprob(theta)

    def optimize(self, quiet=True):
        return optimize.fmin(self.chi2, self.theta_init, disp=not quiet)

    def emcee_traces(self, n_walkers=50, n_trace=200, n_burn=50,
                     theta_start=None, quiet=False, threads=1):
        if theta_start is None:
            theta_start = self.theta_init
        ndim = len(theta_start)
        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.lnprob,
                                        threads=threads)
        start = np.array(theta_start) + 1E-4 * np.random.randn(n_walkers, ndim)

        if not quiet:
            print("Computing {0} traces for N={1} points"
                  ":".format(n_walkers * n_trace,
                             len(self.z)))
            t0 = time.time()
        
        sampler.run_mcmc(start, n_trace, rstate0=np.random.get_state())

        if not quiet:
            print "- Finished in {0:.2g} sec".format(time.time() - t0)

        return sampler.chain[:, n_burn:, :]


class wCDMModel(BaseModel):
    theta_init = [-19.2, 0.28, -1.0]
    Ngrid = 50

    def _eval_single(self, theta, z):
        M0, Om0, w = theta
        z = np.asarray(z)
        cosmo = cosmology.FlatwCDM(H0=70, Om0=Om0, w0=w)
        if z.size > 2 * self.Ngrid:
            zgrid = np.linspace(z.min(), z.max(), self.Ngrid)
            mugrid = cosmo.distmod(zgrid)
            I = interpolate.interp1d(zgrid, mugrid, kind='cubic')
            mu = I(z)
        else:
            mu = cosmo.distmod(z)
        return M0 + mu

    def lnprior(self, theta):
        M0, Om0, w = theta
        if (0.0 <= Om0 <= 1.0) and (-2 <= w <= 0):
            return -(((w + 1) / 0.5) ** 2 +
                     ((Om0 - 0.3) / 0.2) ** 2 +
                     ((M0 + 19.25) / 0.5) ** 2)
        else:
            return -np.inf


class wCDMModel2D(wCDMModel):
    theta_init = [0.3, -1.0]
    M0 = -19.2
    def _eval_single(self, theta, z):
        return wCDMModel._eval_single(self,
                                      [self.M0] + list(theta), z)

    def lnprior(self, theta):
        return wCDMModel.lnprior(self,
                                 [self.M0] + list(theta))


class wCDMModel1D(wCDMModel):
    theta_init = [-1.0]
    M0 = -19.2
    Om = 0.28
    def _eval_single(self, theta, z):
        return wCDMModel._eval_single(self,
                                      [self.M0, self.Om] + list(theta), z)

    def lnprior(self, theta):
        return wCDMModel.lnprior(self,
                                 [self.M0, self.Om] + list(theta))


class QuadraticModel(BaseModel):
    theta_init = [24.9, 2.75, 0.1]
    def _eval_single(self, theta, z):
        return np.polyval(theta[::-1], np.log(z))


class QuadraticModel2D(QuadraticModel):
    theta_init = [2.75, 0.1]
    M0 = 24.9
    def _eval_single(self, theta, z):
        theta = [self.M0] + list(theta)
        return QuadraticModel._eval_single(self, theta, z)
    


def multiscatter(traces, *args, **kwargs):
    ndim = traces.shape[-1]
    traces = traces.reshape(-1, ndim).T
    fig, ax = plt.subplots(ndim, ndim, figsize=(8, 8),
                           sharex='col', sharey='row')

    for i in range(ndim):
        for j in range(ndim):
            ax[i, ndim - 1 - j].plot(traces[j], traces[i], *args, **kwargs)

    return fig, ax


def save_results(Model, z, mag, dmag,
                 Nrange=None, n_walkers=50,
                 n_trace=200, n_burn=0,
                 frames=30):
    output_format = "traces/trace_{model}_{N}.npy"
    if Nrange is None:
        Nrange = np.logspace(1, np.log10(len(z)), frames).astype(int)

    for N in Nrange:
        outfile = output_format.format(model=Model.__name__, N=N)
        if os.path.exists(outfile):
            continue
        model = Model(z[:N], mag[:N], dmag)
        print model.optimize()
        traces = model.emcee_traces(n_walkers=n_walkers,
                                    n_trace=n_trace,
                                    n_burn=n_burn)
        print "- saving to {0}".format(outfile)
        np.save(outfile, traces)


def test_plots(Model=QuadraticModel, datafile='main.dat'):
    z, mag = np.loadtxt(datafile, unpack=True)
    model = Model(z, mag, 0.1)

    # time an evaluation
    print "Timing evaluation on {0} points:".format(len(z))
    t0 = time.time()
    model.eval(model.theta_init, z)
    print "- Finished in {0:.2g} sec".format(time.time() - t0)
    

    # Optimize the model
    print "Performing a MAP optimization:"
    t0 = time.time()
    theta_best = model.optimize()
    print theta_best
    print "- Finished in {0:.2g} sec".format(time.time() - t0)

    # plot the fit
    fig, ax = plt.subplots()
    zfit = np.logspace(-2, np.log10(1.5))
    magfit = model.eval(theta_best, zfit)
    ax.plot(zfit, magfit, '-k')
    ax.plot(z[:1000], mag[:1000], 'ok', ms=5, alpha=0.1)

    # Compute traces
    traces = model.emcee_traces(n_walkers=10, n_trace=200)

    # plot traces
    multiscatter(traces, 'ok', ms=3, alpha=0.05)
    plt.show()


if __name__ == '__main__':
    #test_plots(QuadraticModel2D, 'deep.dat')
    #plt.show()

    z, mag = np.loadtxt('supersample.dat', unpack=True)
    save_results(wCDMModel, z, mag, dmag=0.1,
                 n_walkers=10, n_trace=200)
