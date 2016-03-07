# d-dimensional Gaussian mixture model
import numpy as np
import sys
import matplotlib.pyplot as plt

def eps():
    return 10**-32


def gauss(x, mu, sigma):
    try:
        d = len(x)
    except:
        d = 1
    x = x.reshape((1,d))
    mu = mu.reshape((1,d))
    sigma = sigma.reshape((d,d))
    pi = np.pi
    prec = np.linalg.inv(sigma)
    d = x - mu
    part1 = 1 / ( ((2* pi)**(len(mu)/2)) * (np.linalg.det(sigma)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(prec)).dot((x-mu))

    p = part1 * np.exp(part2)

    return float(p)


def posteriors(self, X):
    try:
        (n,d) = X.shape
    except:
        n = X.shape[0]
        d = 1
    self.N = n
    k = self.ncomponents
    p = np.ones((n, k)) * np.nan
    t = p

    X = X.reshape((n,d))
    self.Mu = self.Mu.reshape((k,d))
    self.Sigma = self.Sigma.reshape((k,d,d))
    self.tau = self.tau.reshape((k,1))

    for iN in range(n):
        for iK in range(k):
            x = np.ones((d,1))*np.nan
            mu = np.ones((d,1))*np.nan
            sigma = np.ones((d,d))*np.nan
            tau = self.tau[iK]
            for iD in range(d):
                x[iD] = X[iN,iD]
                mu[iD] = self.Mu[iK,iD]
                for iD2 in range(d):
                    sigma[iD,iD2] = self.Sigma[iK, iD, iD2]
            p0 = gauss(x, mu, sigma)
            p[iN, iK] = p0 * tau

    return p


def weights(self):
    X = self.X
    p = posteriors(self,X)
    self.post = p
    t = np.nansum(p, axis = 1)
    t = t.reshape((t.size,1))
    self.weight = p / t
    return self


def loglikelihood(self, X):
    tau = self.tau
    p = posteriors(self, X)
    n = self.N
    k = self.ncomponents
    wp = np.ones((n,k))*np.nan
    for iN in range(n):
        for iK in range(k):
            wp[iN,iK] = tau[iK] * p[iN,iK]
    sp = np.sum(wp,axis=1)
    lsp = np.log(sp)

    LnL = np.nansum(lsp)

    return float(LnL)


def expectation(self):
    self = weights(self)

    return self


def maximization(self):
    Nk = np.nansum(self.weight,axis=0)
    tau = Nk / np.nansum(Nk)
    n = self.N
    k = self.ncomponents
    d = self.ndimensions
    X = self.X
    w = self.weight

    X = X.reshape(n,d)
    w = w.reshape(n,k)
    tau = tau.reshape(k,1)
    mu = np.ones((k, d)) * np.nan
    sigma = np.ones((k, d, d)) * np.nan
    for iK in range(k):
        wx = np.ones((n,d))*np.nan
        for iN in range(n):
            for iD in range(d):
                wx[iN,iD] = w[iN,iK]*X[iN,iD]
        sx = np.nansum(wx, axis = 0)
        m = 1 / float(Nk[iK]+eps()) * sx
        mu[iK, :] = m

        D = X - mu[iK,:]
        reg = 1 / float(Nk[iK]+eps())
        wd = np.array([np.dot(D[iN,:],D[iN,:].T)*w[iN,iK] for iN in range(n)])
        s = np.nansum(wd,axis=0)
        sigma[iK, :, :] = reg * s

    I = np.argsort(mu[:,0])
    mu = mu[I,:]
    sigma = sigma[I,:,:]
    tau = tau[I]

    self.Mu = mu
    self.Sigma = sigma
    self.tau = tau

    return self


def evaluation(self):
    X = self.X
    LnL = self.loglikelihood()
    self.LnL = LnL
    return self


def EM(self):
    self=expectation(self)
    self=maximization(self)
    self=evaluation(self)

    return self

def information(self):
    k = self.ncomponents
    N = self.N
    LnL = loglikelihood(self,self.X)
    LnN = np.log(N)
    AIC = 2*k - 2*LnL
    BIC = -2*LnL + k*LnN
    return (AIC,BIC)


def nullModel(self):
    n = self.N
    d = self.ndimensions
    k = self.ncomponents

    X = self.X
    M = np.nanmean(self.X,axis=0)
    M = M.reshape((1,d))
    S = np.zeros((1,d,d))
    for iD1 in range(d):
        for iD2 in range(d):
            x1 = X[:,iD1]
            x2 = X[:,iD2]
            idInf = np.logical_or(np.isinf(x1),np.isinf(x2))
            idNaN = np.logical_or(np.isnan(x1),np.isnan(x2))
            idInvalid = np.logical_or(idInf,idNaN)
            idValid = np.logical_not(idInvalid)
            x1 = x1[idValid]
            x2 = x2[idValid]
            x1 = x1.reshape(x1.size,1)
            x2 = x2.reshape(x2.size,1)
            x0 = np.concatenate((x1,x2),axis=1)
            c = np.cov(x0,rowvar=False)
            S[0,iD1,iD2] = c[1,1]

    null = gmobj(d=self.ndimensions,k=1,mu=M,sigma=S,tau=np.ones((1,d)))
    null.setData(X)
    null.setInformation()

    return null


def seed(self):
    X = self.X
    n = self.N
    d = self.ndimensions
    k = self.ncomponents

    null = nullModel(self)
    mu = null.Mu
    sigma = null.Sigma
    tau = null.tau

    xmax = np.nanmax(X, axis = 0)
    xmin = np.nanmin(X, axis = 0)
    muStart = np.random.random((k, d))
    sigmaStart = np.random.random((k, d, d)) * 2 - 1

    for iK in range(k):
        muStart[iK, :] = muStart[iK,:] * (xmax - xmin) + xmin
        sigmaStart[iK,:,:] = (sigmaStart[iK,:,:]+sigma[0,:,:])**2
        for iD1 in range(d-1):
            for iD2 in range(iD1+1,d):
                s = (sigmaStart[iK,iD1,iD2]+sigmaStart[iK,iD2,iD1])/2
                sigmaStart[iK,iD1,iD2] = s
                sigmaStart[iK,iD2,iD1] = s

    tauStart = np.random.random((k, 1))
    tauStart = tauStart / np.nansum(tauStart)

    boot0 = gmobj(d = d, k = k, mu = muStart, sigma = sigmaStart, tau = tauStart)
    boot0.X = X
    boot0.N = n
    boot0.post = np.ones((n, k), float) * np.nan
    boot0.weight = np.ones((n, k), float) * np.nan
    boot0.LnL = -np.inf
    LnL0 = null.loglikelihood()
    (AIC0,BIC0) = null.information()
    boot0.LnL0 = LnL0
    boot0.AIC0 = AIC0
    boot0.BIC0 = BIC0

    return boot0


class gmobj(object):
    """
    Gaussian mixture object class
    """

    def posteriors(self,X):
        """
        Method to calculate the posterior probability of the data in X.

        :param X: N x d matrix of N data points measured in d dimensions
        :return: N x K matrix of N posterior probabilities of each of the K components
        """
        try:
            (n,d) = X.shape
        except:
            n = X.shape[0]
            d = 1

        assert d==self.ndimensions, 'X must be n x '+str(self.ndimensions)+'-dimensional matrix.'

        p = posteriors(self,X)
        return p


    def information(self):
        """
        Method to evaluate the information criteria of the current gmobj.

        :return: Tuple with
            (AIC, BIC)
            where
                AIC         is the Akaike information criterion calculated as 2*k - 2*Ln[Likelihood]
                BIC         is the Bayesian information criterion calculated as Ln[N]*k - 2*Ln[Likelihood]
        """
        (AIC,BIC) = information(self)
        return (AIC,BIC)


    def loglikelihood(self):
        """
        Method to evaluate the log-likelihood of the current gmobj.

        :return: LnL
        """
        X = self.X
        LnL = loglikelihood(self,X)
        return LnL


    def nullModel(self):
        """
        Method to evaluate the null (single-Gaussian non-mixture) model of the current gmobj.

        :return: null gmobj
        """
        null = nullModel(self)
        return null


    def setMu(self,Mu):
        """
        Method to set the mean attribute.

        :param Mu: K x d matrix of means setting m[d=j] for component k.
        """
        try:
            (k,d) = Mu.shape
        except:
            k = Mu.shape[0]
            d = 1
        self.ncomponents = k
        self.Mu = mu

    def setSigma(self,Sigma):
        """
        Method to set the covariance matrix attribute

        :param Sigma: K x d x d matrix of covariance matrices setting Cov[Xi,Xj] for component k.
        """
        try:
            (k,d1,d2) = Sigma.shape
        except:
            try:
                (d1,d2) = Sigma.shape
                k = 1
            except:
                d2 = Sigma.shape[0]
                d1 = 1
                k = 1
        assert d1==d2, 'Sigma must be k x d x d'

        Sigma = Sigma.reshape(k,d,d)
        self.ncomponents = k
        self.ndimensions = d
        self.Sigma = Sigma

    def setTau(self,tau):
        """
        Method to set the mixing proportion attribute

        :param tau: K x 1 vector of mixing proportions setting P[component=k]
        """
        tau = tau.reshape(tau.size,1)
        k = tau.shape[0]
        self.ncomponents = k
        self.tau = tau

    def setData(self,X):
        """
        Method to set the data attribute for gmobj.

        :param X: N x d matrix of N data points measured in d dimensions.
        """
        try:
            (n,d) = X.shape
        except:
            n = X.shape[0]
            d = 1

        k = self.ncomponents
        self.N = n
        self.ndimensions = d
        self.X = X
        self.post = np.ones((n, k), float) * np.nan
        self.weight = np.ones((n, k), float) * np.nan


    def setInformation(self):
        """
        Method to set the information attributes LnL, AIC, and BIC.

        """
        (AIC,BIC) = self.information()
        self.LnL = self.loglikelihood()
        self.BIC = BIC
        self.AIC = AIC


    def setNullModel(self):
        """
        Method to set the null model attributes LnL0, AIC0, and BIC0.


        """
        null = self.nullModel()
        LnL0 = null.loglikelihood()
        (AIC0,BIC0) = null.information()
        self.LnL0 = LnL0
        self.AIC0 = AIC0
        self.BIC0 = BIC0


    def parseDistributions(self):
        """
        Parses gmobj into its component multivariate Gaussians.

        :return: 1 x ncomponents list of dictionnaries with keys
          'Mu': Mean of component in each dimension
          'Sigma': Covariance matrix of component
          'Tau': Mixture proportion of component
        """
        k = self.ncomponents
        d = self.ndimensions
        GMDistributionList = list()
        for iK in range(k):
            Mu = self.Mu[iK,:]
            Sigma = (self.Sigma[iK,:,:]).reshape(d,d)
            Tau = self.tau[iK]
            GMDistribution = {'Mu':Mu,'Sigma':Sigma,'Tau':Tau}
            GMDistributionList.append(GMDistribution)
        return GMDistributionList

    def getIterHist(self):
        """
        Returns the iteration history of the current fit

        :return:
        """
        iterHist = self.IterHist
        return iterHist


    def plotIters(self):
        """
        Plots the iteration history of Ln[L].


        """
        iters = self.IterHist
        niters = len(iters)
        history = np.arange(niters)+1
        fh=plt.figure()
        ah=fh.add_axes()
        ph=plt.plot(history,iters)
        plt.xlabel('Iteration')
        plt.ylabel('Ln[L]')
        ah.set_title('Iteration history')

        return (fh, ah, ph)

    def plot(self):
        """
        Method to plot 1D gaussians along each dimension

        :return: List of (fh, ah, [ph]) handles.
        """
        d = self.ndimensions
        k = self.ncomponents
        Mu = self.Mu
        Sigma = self.Sigma
        tau = self.tau
        try:
            X = self.X
            n = self.N
            X = X.reshape(n,d)

            xmin = np.nanmin(X,axis=0)
            xmax = np.nanmax(X,axis=0)
        except:
            print('Warning: Data not yet associated with gmobj. Using dummy data.')
            n = 0
            s = np.ones((1,d))
            for iD in range(d):
                s0 = Sigma[:,iD,iD]
                s[iD] = np.sqrt(np.max(s0))
            xmin = np.nanmin(Mu,axis=0)-s
            xmax = np.nanmax(Mu,axis=0)+s

        X = X.reshape(n,d)
        cmap = np.linspace(0.2,0.8,k)

        out = list()
        for iD in range(d):
            fh = plt.figure()
            ah=fh.add_axes()
            ph = list()
            if n>0:
                (f,bin)=np.histogram(X[:,iD],np.ceil(np.sqrt(n))+1)
                binc = np.array([(bin[ix]+bin[ix+1])/2 for ix in range(len(bin)-1)])
                ph0=plt.plot(binc,f/np.nansum(f),'k-',linewidth=3)
                x1 = np.linspace(xmin[iD],xmax[iD],num=n)
                n1 = n
                ph.append(ph0)
            else:
                x1 = np.linspace(xmin[iD],xmax[iD],num=500)
                n1 = 500
            for iK in range(k):
                y1 = np.array([gauss(x1[iN],Mu[iK,iD],Sigma[iK,iD,iD])*tau[iK] for iN in range(n1)])
                ph0=plt.plot(x1,y1,str(cmap[iK]))
                ph.append(ph0)
            out.append((fh,ah,ph))

        return out

    def plot2d(self,dims):
        """
        Method to plot contour of 2 dimensions.

        :return: Tuple of figure handles
        """
        assert len(dims)==2, 'Only 2 dimensions can be plotted as a contour with this method.'
        X = self.X
        xmin = np.nanmin(self.X[:,dims],axis=0)
        xmax = np.nanmax(self.X[:,dims],axis=0)

        xmean = np.nanmean(self.X,axis=0)
        lx = linspace(xmin[dims[0]],xmax[dims[0]],100)
        ly = linspace(xmin[dims[1]],xmax[dims[1]],100)

        XY = np.ones((100,d))

        GMList = self.parseDistributions()
        k = len(GMList)

        z = np.ones((100,100))*np.nan
        for ix in range(100):
            for iy in range(100):
                p = 0
                for ik in range(k):
                    x = xmean
                    x[dims[0]] = lx[ix]
                    x[dims[1]] = ly[ix]
                    gms = GMlist[ik]
                    mu = gms['Mu']
                    sigma = gms['Sigma']
                    tau = gms['Tau']
                    p = p + gauss(x,mu,sigma)*tau
                z[iy,ix] = p

        fh=plt.figure()
        ah=fh.add_axes()
        ph1=plt.imshow(z,vmin=0,vmax=np.nanmax(z))
        plt.xlabel('Col'+str(dims[0]))
        plt.ylabel('Col'+str(dims[1]))
        ph2=plt.scatter(X[:,dims[0]],X[:,dims[1]],marker='.')
        return (fh,ah,(ph1,ph2))


    def __init__(self, d = None, k = None, mu = None, sigma = None, tau = None):
        """
        Creates an instance of the gmobj class.

        Optional:
        :param d: Dimensionality of feature set (d-dimensional x)
        :param k: Number of mixture components (k-gaussian mixture)
        :param mu: k x d matrix of means, with mu[i,:] the mean of component i for each dimension d
        :param sigma: k x d x d matrix of covariance
        :param tau: k x 1 vector of mixture proportions, with tau[i] = P[component = i]

        :return: gmobj with attributes
            gmobj.ndimensions = d
            gmobj.ncomponents = k
            gmobj.Mu = mu
            gmobj.Sigma = sigma
            gmobj.tau = tau
        """
        self.ndimensions = d
        self.ncomponents = k
        self.Mu = mu
        self.Sigma = sigma
        self.tau = tau

        self.X = np.array((0))
        self.N = 0
        self.post = np.array((0))
        self.weight = np.array((0))
        self.LnL = np.nan
        self.AIC = np.nan
        self.BIC = np.nan
        self.LnL0 = np.nan
        self.AIC0 = np.nan
        self.BIC0 = np.nan


    def __str__(self):
        d = self.ndimensions
        k = self.ncomponents
        tau = self.tau
        if d is None: d=0
        if k is None: k=0
        if tau is None: tau=np.array(0)
        string1 = 'gmobj with {0:d} components in {1:d} dimensions:\n'.format(k,d)
        string2 = ''
        for iK in range(k):
            string2 = string2+'Component {0:d}: {1:.2f}%, ['.format(iK+1, float(tau[iK])*100)
            for iD in range(d-1):
                string2 = string2+'{0:.3f}, '.format(self.Mu[iK,iD])
            string2 = string2+'{0:.3f}]\n'.format(self.Mu[iK,self.ndimensions-1])
        string2 = string2+'\n'
        string3 = 'Ln[L] = '+str(self.LnL)+'\n'
        string3 = 'Ln[L0] = '+str(self.LnL0)+'\n'+string3
        string4 = 'AIC/AIC0 = {0:.3f}/{1:.3f} = {2:.3f}\n'.format(self.AIC,self.AIC0,self.AIC/self.AIC0)
        string5 = 'BIC/BIC0 = {0:.3f}/{1:.3f} = {2:.3f}\n'.format(self.BIC,self.BIC0,self.BIC/self.BIC0)
        return string1+string2+string3+string4+string5

    def __eq__(self,other):
        test = True
        if self.ncomponents == other.ncomponents and self.ndimensions == other.ndimensions:
            k = self.ncomponents
            d = self.ndimensions
            for iK in range(k):
                test = test and self.tau[iK]==other.tau[iK]
                for iD in range(d):
                    test = test and self.Mu[iK,iD]==other.Mu[iK,iD]
                    for iD2 in range(d):
                        test = test and self.Sigma[iK,iD,iD2]==other.Mu[iK,iD,iD2]
            return test
        else:
            test = False
        return test

    def __lt__(self,other):
        return self.LnL < other.LnL


    def fit(self, X, k, maxiter = 500, tolfun = 10**-16, nrep = 1000):
        """
        Fits a Gaussian mixture model to the data in X using k components.

        :param X: n x d matrix of features
        :param k: Number of components in Gaussian mixture

        Optional:
        :param maxiter=500: maximum number of iterations before non-convergence
        :param tolfun=0.001: smallest change in log-likelihood before convergence
        :param nrep=1000: number of random start points to use
        :return: gmobj with attributes

        """
        try:
            (n, d) = X.shape
        except:
            n = X.shape[0]
            d = 1
        X = X.reshape(n,d)

        self = gmobj(d=d,k=k)
        self.setData(X)
        self.ndimensions = d
        self.ncomponents = k

        print('Evaluating null model...')
        self.setNullModel()

        boot = list()
        LnLs = list()
        print('\n')
        iterHist = np.ones((maxiter,nrep))*np.nan
        attempt = 1
        print('Evaluating Gaussian Mixture Distribution object...')
        for rep in range(nrep):
            boot0 = seed(self)
            lastLnL = -np.inf
            delta = -np.inf
            iter = 0
            while True:
                if (attempt % 100) == 0:
                    print('.', end = '\n')
                else:
                    print('.', end = '')
                sys.stdout.flush()
                iter = iter + 1
                boot0 = EM(boot0)
                delta = lastLnL - boot0.LnL
                iterHist[iter-2,rep] = boot0.LnL
                lastLnL = boot0.LnL
                if iter > maxiter: break
                attempt += 1
                #if delta < tolfun: break
            print('\n')
            sys.stdout.flush()
            boot.append(boot0)
            LnLs.append(boot0.LnL)
        print('\n')
        print('Found best', str(k), 'components in', str(d), '-dimensional data.')
        I = np.argmax(LnLs)
        fitgm = boot[I]

        self.X = fitgm.X
        self.ndimensions = fitgm.ndimensions
        self.ncomponents = fitgm.ncomponents
        self.N = fitgm.N
        self.post = fitgm.post
        self.weight = fitgm.weight
        self.Mu = fitgm.Mu
        self.Sigma = fitgm.Sigma
        self.tau = fitgm.tau
        self.LnL = fitgm.LnL
        self.IterHist = iterHist[:,I]

        print('Evaluating information criteria of the fit...')
        self.setInformation()

        print(self)
        return self

