# d-dimensional Gaussian mixture model
import numpy as np
import scipy.stats as stats
import sys
import time
import matplotlib.pyplot as plt


def eps():
    return 10**-32


def covariance(X):
    try:
        (n,d) = X.shape
    except:
        n = X.shape[0]
        d = 1

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
            S[0,iD1,iD2] = c[0,1]
    return S

def gauss(x, mu, sigma):
    try:
        d = len(x)
    except:
        d = 1

    pi = np.pi

    if d>1:
        x = np.array(x)
        mu = np.array(mu)
        sigma = np.array(sigma)
    else:
        x = np.array([x])
        mu = np.array([mu])
        sigma = np.array([sigma])

    x = x.reshape(d,1)
    mu = mu.reshape(d,1)
    sigma = sigma.reshape(d,d)

    if d>1:
        prec = np.linalg.inv(sigma+eps())
    else:
        prec = 1/(sigma+eps())

    dev = x - mu

    covDet = np.linalg.det(sigma)

    denom = ( ((2* pi)**(d/2)) * (covDet**(1/2)) )

    part1 = 1 / denom
    part2 = (-1/2) * (dev.T.dot(prec).dot(dev))
    p = part1 * np.exp(part2)

    #mvn = stats.multivariate_normal(mean=mu,cov=sigma)
    #p = mvn.pdf(x)

    return float(p)


def posteriors(gmobj, X):
    try:
        (n,d) = X.shape
    except:
        n = X.shape[0]
        d = 1
    gmobj.N = n
    k = gmobj.getKcomponents()
    p = np.ones((n, k)) * np.nan
    t = np.ones((n, k)) * np.nan

    X = X.reshape((n,d))
    Mu = (gmobj.getMu()).reshape((k,d))
    Sigma = (gmobj.getSigma()).reshape((k,d,d))
    Tau = (gmobj.getTau()).reshape((k,1))

    for iN in range(n):
        for iK in range(k):
            x = np.ones((d,1))*np.nan
            mu = np.ones((d,1))*np.nan
            sigma = np.ones((d,d))*np.nan
            tau = Tau[iK]
            for iD in range(d):
                x[iD] = X[iN,iD]
                mu[iD] = Mu[iK,iD]
                for iD2 in range(d):
                    sigma[iD,iD2] = Sigma[iK, iD, iD2]
            p0 = gauss(x, mu, sigma)
            p[iN, iK] = p0 * tau

    return p


def weights(gmobj):
    X = gmobj.X
    p = posteriors(gmobj,X)
    gmobj.post = p
    t = np.nansum(p, axis = 1)
    t = t.reshape((t.size,1))
    weight = p / t
    return weight


def loglikelihood(gmobj, X):
    tau = gmobj.getTau()
    p = posteriors(gmobj, X)
    n = gmobj.getN()
    k = gmobj.getKcomponents()

    wp = np.ones((n,k))*np.nan
    for iN in range(n):
        for iK in range(k):
            wp[iN,iK] = tau[iK] * p[iN,iK]
    sp = np.sum(wp,axis=1)
    lsp = np.log(sp)

    LnL = np.nansum(lsp)

    return float(LnL)


def expectation(self):
    self.weight = weights(self)

    return self


def maximization(gmobj):
    w = gmobj.weight
    Nk = np.nansum(w,axis=0)
    tau = Nk / np.nansum(Nk)
    n = gmobj.getN()
    k = gmobj.getKcomponents()
    d = gmobj.getDdimensions()
    X = gmobj.getData()

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
        wd = np.ones((n,d,d))*np.nan
        for iN in range(n):
            for iD in range(d):
                wd[iN,iD,:] = D[iN,iD]*D[iN,:]*w[iN,iK]
        s = np.nansum(wd,axis=0)
        sigma[iK, :, :] = reg * s

    I = np.argsort(mu[:,0])
    mu = mu[I,:]
    sigma = sigma[I,:,:]
    tau = tau[I]

    gmobj.setMu(mu)
    gmobj.setSigma(sigma)
    gmobj.setTau(tau)

    return gmobj


def evaluation(gmobj):
    X = gmobj.getData()
    LnL = gmobj.loglikelihood()
    gmobj.LnL = LnL
    return gmobj


def EM(gmobj):
    gmobj=expectation(gmobj)
    gmobj=maximization(gmobj)
    gmobj=evaluation(gmobj)

    return gmobj

def information(gmobj):
    k = gmobj.getKcomponents()
    N = gmobj.getN()
    LnL = loglikelihood(gmobj,gmobj.getData())
    LnN = np.log(N)
    AIC = 2*k - 2*LnL
    BIC = -2*LnL + k*LnN
    return (AIC,BIC)


def nullModel(self):
    n = self.getN()
    d = self.getDdimensions()
    k = self.getKcomponents()

    X = self.getData()
    M = np.nanmean(self.X,axis=0)
    M = M.reshape((1,d))
    S = covariance(X)
    null = GMobj(d=self.ndimensions,k=1,mu=M,sigma=S,tau=1)
    null.setData(X)
    null.setInformation()

    return null


def seed(self):
    X = self.getData()
    n = self.getN()
    d = self.getDdimensions()
    k = self.getKcomponents()

    null = nullModel(self)

    sigma = null.getSigma()

    xmax = np.nanmax(X, axis = 0)
    xmin = np.nanmin(X, axis = 0)
    muStart = np.random.random((k, d))
    # sigmaStart = np.random.random((k, d, d)) * 2 - 1

    sigmaStart = np.zeros((k,d,d))
    for iK in range(k):
        sigmaStart[iK,:,:] = sigma[:,:]/k

    tauStart = np.random.random((k, 1))
    tauStart = tauStart / np.nansum(tauStart)

    boot0 = GMobj(d = d, k = k, mu = muStart, sigma = sigmaStart, tau = tauStart)
    boot0.setData(X)
    boot0.setInformation()
    boot0.setLnL()
    boot0.LnL = -np.inf

    return boot0


class GMobj(object):
    """
    Gaussian mixture object class:

    SET methods:
    ************
    gmobj.setKcomponents(k) sets the number of components in gaussian mixture gmobj to k.
    gmobj.setDdimensions(d) sets the dimensionality of the gaussian mixture gmobj to d.
    gmobj.setMu([[Mu_11,Mu_12,...,Mu_1d],...,[Mu_k1,Mu_k2,...,Mu_kd]]) sets the mean of each component along each dimension of gaussian mixture gmobj.
    gmobj.setSigma([CovMat_1],...,[CovMat_k]) sets the dimension-by-dimension covariance matrix of each component of gaussian mixture gmobj.
    gmobj.setTau([Tau_1],...,[Tau_k]) sets the mixing proportion of each component of gaussian mixture gmobj.
    gmobj.setData(X) sets the data of gaussian mixture gmobj to the n x d matrix X.

    GET methods:
    ************
    p = gmobj.posteriors(X) returns the posterior probability of X given gaussian mixture gmobj.
    fitObj = gmobj.fit(X,k) returns a gaussian mixture fitObj fit by Expectation-Maximization to the data in X, assuming k components.

    k = gmobj.getKcomponents() returns the number of components for gaussian mixture gmobj.
    d = gmobj.getDdimensions() returns the dimensionality of the gaussian mixture gmobj.
    Mu = gmobj.getMu() returns a k x d matrix of means of each component along each dimension of gaussian mixture gmobj.
    Sigma = gmobj.getSigma() returns a k x d x d matrix of variance-covariance matrices for each component of gaussian mixture gmobj.
    Tau = gmobj.getTau() returns a k x 1 vector of mixing proportions of each component of gaussian mixture gmobj.
    X = gmobj.getData() returns a n x d matrix of data for gaussian mixture gmobj.
    N = gmobj.getN() returns an int of the number of case observations in the data of gaussian mixture gmobj.
    IterHist = gmobj.getIterHist() returns a list of LnL values calculated at each iteration step of the Expectation-Maximization algorithm used in the fit of gaussian mixture object gmobj.
    LnL = gmobj.loglikelihood() returns the loglikelihood of data in gaussian mixture gmobj.
    (AIC, BIC) = gmobj.getInformation() returns information criteria of gaussian mixture gmobj.
    null = gmobj.nullModel() returns a null-model (single-component) gaussian mixture object of the data in gmobj.

    DISPLAY methods:
    ****************
    gmList = gmobj.parseDistributions() returns a len-k list of dictionaries for each component with keys
        'Mu', a 1xd vector of means for the component along each dimension
        'Sigma', a d x d variance-covariance matrix for the component
        'Tau', a float with the mixing proportion of the component

    gmobj.plotIters() plots a figure of the LnL calculated at each iteration step vs. the iteration step
    gmobj.plot() creates d plots of
        the relative frequency polygons of the data along each dimension (thick black line), and
        each of the k gaussian probability density functions (thin grey lines) along that dimension,
        for the gaussian mixture gmobj.
    gmobj.plot2d(dims=[0,1]) creates a single filled contour plot of the multivariate density of the mixture along specified dimensions, and a scatter of the observed data points in gaussian mixture gmobj.


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

        assert d==self.getDdimensions(), 'X must be n x '+str(self.getDdimensions())+'-dimensional matrix.'

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
        X = self.getData()
        LnL = loglikelihood(self,X)
        return LnL

    def nullModel(self):
        """
        Method to evaluate the null (single-Gaussian non-mixture) model of the current gmobj.

        :return: null gmobj
        """
        null = nullModel(self)
        return null

    def setLnL(self):
        """
        Method to set the LnL and null-model LnL0 attribute of the current gmobj.

        :return:
        """
        self.LnL = self.loglikelihood()
        k = self.getKcomponents()
        if k==1:
            self.LnL0 = self.loglikelihood()
        else:
            self.LnL0 = (self.nullModel()).loglikelihood()

    def setInformation(self):
        """
        Method to set the information attributes AIC and BIC, and the null-model information attributes AIC0 and BIC0.

        :return:
        """
        (self.AIC, self.BIC) = self.information()
        k = self.getKcomponents()
        if k==1:
            (self.AIC0, self.BIC0) = self.information()
        else:
            (self.AIC0, self.BIC0) = (self.nullModel()).information()

    def getInformation(self):
        """
        Method to return a tuple of information criteria.

        :return:
        (AIC, BIC)
        """
        return (self.AIC,self.BIC)

    def setKcomponents(self,k):
        """
        Method to set the number of components in the gmobj

        :param k: number of components
        :return:
        """
        self.kcomponents = k

    def getKcomponents(self):
        """
        Method to return the number of components in the gmobj

        :return:
        """
        return self.kcomponents

    def setDdimensions(self,d):
        """
        Method to set the number of dimensions in the gmobj

        :param d: number of dimensions
        :return:
        """
        self.ddimensions = d

    def getDdimensions(self):
        """
        Method to return the number of dimensions in the gmobj

        :return:
        """
        return self.ddimensions

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
        self.setKcomponents(k)
        self.Mu = Mu

    def getMu(self):
        """
        Method to return the mean of the gmobj

        :return: k x d matrix of means
        """
        k = self.getKcomponents()
        d = self.getDdimensions()
        mu = self.Mu
        return mu.reshape(k,d)

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
        d = d1

        Sigma = Sigma.reshape(k,d,d)
        self.setKcomponents(k)
        self.setDdimensions(d)

        self.Sigma = Sigma

    def getSigma(self):
        k = self.getKcomponents()
        d = self.getDdimensions()
        sigma = self.Sigma
        return sigma.reshape(k,d,d)

    def setTau(self,tau):
        """
        Method to set the mixing proportion attribute

        :param tau: K x 1 vector of mixing proportions setting P[component=k]
        """
        tau = tau.reshape(tau.size,1)
        k = tau.shape[0]
        self.setKcomponents(k)
        self.Tau = tau/np.nansum(tau)


    def getTau(self):
        """
        Method to return the mixing proportion attribute of the gmobj

        :return:
        """
        tau = self.Tau
        return tau/np.nansum(tau)

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

        self.X = X
        self.N = n
        self.setDdimensions(d)

        k = self.getKcomponents()
        if (np.isnan(self.getMu())).all():
            mu = np.nanmean(X,axis=0)
            Mu = np.ones((k,d))*np.nan
            for iK in range(k):
                Mu[iK,:] = mu

            self.setMu(Mu)

        if (np.isnan(self.getSigma())).all():
            sigma = covariance(X)
            Sigma = np.ones((k,d,d))*np.nan
            for iK in range(k):
                Sigma[iK,:,:] = sigma

            self.setSigma(Sigma)

        if (np.isnan(self.getTau())).all():
            tau = np.ones((k,1))*(1/k)
            self.setTau(tau)

        self.post = self.posteriors(X)
        self.weight = weights(self)

    def getData(self):
        """
        Method to return the data attribute for gmobj.

        :return:
        """
        return self.X

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
        k = self.getKcomponents()
        d = self.getDdimensions()
        GMDistributionList = list()
        for iK in range(k):
            Mu = self.getMu()[iK,:]
            Sigma = (self.getSigma()[iK,:,:]).reshape(d,d)
            Tau = self.getTau()[iK]
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

    def getN(self):
        return self.N

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
        plt.title('Iteration history')

        return (fh, ah, ph)

    def plot(self):
        """
        Method to plot 1D gaussians along each dimension

        :return: List of (fh, ah, [ph]) handles.
        """
        d = self.getDdimensions()
        k = self.getKcomponents()
        Mu = self.getMu()
        Sigma = self.getSigma()
        tau = self.getTau()
        try:
            X = self.getData()
            n = self.getN()
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
            xmin = np.nanmin(Mu,axis=0)-3*s
            xmax = np.nanmax(Mu,axis=0)+3*s

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

    def plot2d(self,dims=[0,1]):
        """
        Method to plot contour of 2 dimensions.

        :param: dims=[0,1]: dimensions of data in gmobj for which to plot contour.
        :return: Tuple of figure handles
        """
        assert len(dims)==2, 'Only 2 dimensions can be plotted as a contour with this method.'
        X = self.getData()

        xmin = np.nanmin(self.X[:,dims],axis=0)
        xmax = np.nanmax(self.X[:,dims],axis=0)

        xmean = np.nanmean(self.X,axis=0)
        lx = np.linspace(xmin[dims[0]],xmax[dims[0]],100)
        ly = np.linspace(xmin[dims[1]],xmax[dims[1]],100)
        lx = lx.reshape(1,len(lx))
        ly = ly.reshape(1,len(ly))

        XX = np.ones((100,100))
        YY = np.ones((100,100))
        for iX in range(100):
            XX[iX,:] = lx
        for iY in range(100):
            YY[iY,:] = ly
        YY = YY.T

        GMList = self.parseDistributions()
        k = len(GMList)

        z = np.ones((100,100))*np.nan
        for ix in range(100):
            for iy in range(100):
                p = 0
                for ik in range(k):
                    x = xmean
                    x[dims[0]] = XX[iy,ix]
                    x[dims[1]] = YY[iy,ix]

                    gms = GMList[ik]
                    mu = gms['Mu']
                    sigma = gms['Sigma']
                    tau = gms['Tau']
                    p = p + gauss(x,mu,sigma)*tau
                z[iy,ix] = p

        fh=plt.figure()
        ah=fh.add_axes()
        ph1=plt.contourf(XX,YY,z,vmin=0)
        plt.xlabel('Col'+str(dims[0]))
        plt.ylabel('Col'+str(dims[1]))
        ph2=plt.scatter(X[:,dims[0]],X[:,dims[1]],marker='.')
        plt.xlim(np.nanmin(XX),np.nanmax(XX))
        plt.ylim(np.nanmin(YY),np.nanmax(YY))
        plt.colorbar(mappable=ph1)
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
            gmobj.Tau = tau
        """
        self.ndimensions = d
        self.ncomponents = k

        if k is None:
            k = 0
        if d is None:
            d = 0

        if mu is None:
            mu = np.ones((k,d))*np.nan
        if sigma is None:
            sigma = np.ones((k,d,d))*np.nan
        if tau is None:
            tau = np.ones((k,1))*np.nan
        if k<2:
            tau = np.ones((k,1))

        assert mu.size == k*d, 'mu must be Kcomponents x Ddimensions matrix of means'
        assert sigma.size == k*d*d, 'sigma must be Kcomponents x Ddimensions x Ddimensions matrix of covariance matrices'
        assert tau.size == k, 'tau must be Kcomponents x 1 vector of mixture proportions'

        self.setMu((np.array(mu)).reshape(k,d))
        self.setSigma((np.array(sigma)).reshape(k,d,d))
        self.setTau(np.array(tau).reshape(k,1))

        self.LnL = np.nan
        self.AIC = np.nan
        self.BIC = np.nan
        self.LnL0 = np.nan
        self.AIC0 = np.nan
        self.BIC0 = np.nan


    def __str__(self):
        d = self.getDdimensions()
        k = self.getKcomponents()
        tau = self.getTau()
        if d is None: d=0
        if k is None: k=0
        if tau is None: tau=np.array(0)
        string1 = 'gmobj with {0:d} components in {1:d} dimensions:\n'.format(k,d)
        string2 = ''
        for iK in range(k):
            string2 = string2+'Component {0:d}: {1:.2f}%, ['.format(iK+1, float(tau[iK])*100)
            for iD in range(d-1):
                string2 = string2+'{0:.3f}, '.format(self.getMu()[iK,iD])
            string2 = string2+'{0:.3f}]\n'.format(self.getMu()[iK,self.ndimensions-1])
        string2 = string2+'\n'
        string3 = 'Ln[L] = '+str(self.LnL)+'\n'
        string3 = 'Ln[L0] = '+str(self.LnL0)+'\n'+string3
        (AIC,BIC) = self.information()
        (AIC0,BIC0) = (self.nullModel()).information()


        string4 = 'AIC/AIC0 = {0:.3f}/{1:.3f} : {2:.3f} ln relative likelihood of model\n'.format(AIC,AIC0,(AIC-AIC0)/2)
        string5 = 'BIC/BIC0 = {0:.3f}/{1:.3f} : {2:.3f} ln relative likelihood of model\n'.format(BIC,BIC0,(BIC-BIC0)/2)
        return string1+string2+string3+string4+string5

    def __repr__(self):
        d = self.getDdimensions()
        k = self.getKcomponents()
        GMList = self.parseDistributions()
        str1 = '<{0:d} component, {1:d}-dimensional gaussian mixture object>\n'.format(k,d)
        for iK in range(k):
            gm = GMList[iK]
            str2 = 'Component {0:d}:\nTau = {1:.4f}\n'.format(iK+1,float(gm['Tau']))
            str3 = 'mu = ['
            for iD in range(d-1):
                str3 = str3 + '{0:d}, '.format(gm['Mu'][iD])
            str3 = str3 + '{0:d}]\n'.format(gm['Mu'][d])
            str4 = 'sigma = '+str(gm['Sigma'])+'\n'


            str1 = str1 + str2 + str3 + str4
        return str1


    def __eq__(self,other):
        test = True
        if self.getKcomponents() == other.getKcomponents() and self.getDdimensions() == other.getDdimensions():
            k = self.getKcomponents()
            d = self.getDdimensions()
            for iK in range(k):
                test = test and self.getTau()[iK]==other.getTau()[iK]
                for iD in range(d):
                    test = test and self.getMu()[iK,iD]==other.getMu()[iK,iD]
                    for iD2 in range(d):
                        test = test and self.getSigma()[iK,iD,iD2]==other.getSigma()[iK,iD,iD2]
            return test
        else:
            test = False
        return test

    def __lt__(self,other):
        return self.LnL < other.LnL

    def __add__(self,other):
        N1 = self.getN()
        d1 = self.getDdimensions()
        k1 = self.getKcomponents()
        Mu1 = self.getMu()
        Sigma1 = self.getSigma()
        Tau1 = self.getTau()

        N2 = other.getN()
        d2 = other.getDdimensions()
        k2 = other.getKcomponents()
        Mu2 = other.getMu()
        Sigma2 = other.getSigma()
        Tau2 = other.getTau()

        assert d1 == d2, "Gaussian mixture objects must have equal dimensionality to be added."

        try:
            X1 = self.X
        except:
            X1 = np.ones((N1,d1))*np.nan
        try:
            X2 = other.X
        except:
            X2 = np.ones((N2,d2))*np.nan

        N = N1+N2
        d = (d1+d2)/2
        k = k1+k2

        if N>0:
            X = np.concatenate((X1,X2),axis=0)
        else:
            X = np.ones((N,d))*np.nan

        Mu = np.ones((k,d))*np.nan
        Sigma = np.ones((k,d,d))*np.nan
        Tau = np.ones((k,1))*np.nan
        for iK1 in range(k1):
            Mu[iK1,:] = Mu1[iK1,:]
            Sigma[iK1,:,:] = Sigma1[iK1,:,:]
            Tau[iK1] = Tau1[iK1]
        for iK2 in range(k2):
            Mu[iK2,:] = Mu2[iK2,:]
            Sigma[iK2,:,:] = Sigma2[iK2,:,:]
            Tau[iK2] = Tau2[iK2]

        I = np.argsort(Mu[:,0])
        Mu = Mu[I,:]
        Sigma = Sigma[I,:,:]
        Tau = Tau[I]

        aug = GMobj(d=d,k=k,mu=Mu,sigma=Sigma,tau=Tau)
        aug.setData(X)
        return aug


    def fit(self, X, k, miniter=10, maxiter = 100, tolfun = 10**-32, nrep = 50):
        """
        Fits a Gaussian mixture model to the data in X using k components.

        :param X: n x d matrix of features
        :param k: Number of components in Gaussian mixture

        Optional:
        :param maxiter=1000: maximum number of iterations before non-convergence
        :param tolfun=0.001: smallest change in log-likelihood before convergence
        :param nrep=500: number of random start points to use
        :return: gmobj with attributes

        """
        try:
            (n, d) = X.shape
        except:
            n = X.shape[0]
            d = 1
        X = X.reshape(n,d)

        self = GMobj(d=d,k=k)
        self.setData(X)

        print('Evaluating null model...')
        self.setNullModel()

        boot = list()
        LnLs = list()
        iterHist = np.ones((maxiter,nrep))*np.nan
        attempt = 1
        print('Evaluating Gaussian Mixture Distribution object...')
        start = time.clock()
        remTime = False
        for rep in range(nrep):
            boot0 = seed(self)
            lastLnL = -np.inf
            delta = -np.inf
            iter = 0
            while True:
                if iter > maxiter: break
                if iter > miniter and delta < tolfun: break

                sys.stdout.flush()
                iter = iter + 1
                boot0 = EM(boot0)
                delta = np.abs(lastLnL - boot0.LnL)
                iterHist[iter-2,rep] = boot0.LnL
                lastLnL = boot0.LnL

                if attempt==1:
                    end = time.clock()
                    print('One iteration took',end-start,'sec.')
                    print('Minimum time:     ',((end-start)*miniter*nrep)/60,'minutes.')
                    print('Maximum time:     ',((end-start)*maxiter*nrep)/60,'minutes.')
                if (attempt % 100) == 0:
                    print('.', end = '\n')
                else:
                    print('.', end = '')
                if (attempt % 100) == 0 and remTime==False:
                    end = time.clock()
                    # how many replications are left?
                    repLeft=(nrep-rep)
                    if repLeft<nrep:
                        # how many iterations are done per rep?
                        it = np.logical_not(np.isnan(iterHist[:,:rep]))
                        iPerRep = np.nanmean(np.nansum(it,axis=0))
                        print('Est. remaining:    ',(((end-start)/attempt)*(iPerRep*(repLeft-1)+(iPerRep-iter)))/60,'minutes.')
                        remTime=True

                attempt += 1

            sys.stdout.flush()
            boot.append(boot0)
            LnLs.append(boot0.LnL)
        print('\n')
        print('Found best', str(k), 'components in', str(d), '-dimensional data.')
        I = np.argmax(LnLs)
        fitgm = boot[I]

        del self

        self = GMobj(d=fitgm.getDdimensions(),k=fitgm.getKcomponents(),mu=fitgm.getMu(),sigma=fitgm.getSigma(),tau=fitgm.getTau())

        self.setData(X)

        self.setLnL()

        self.IterHist = iterHist[:,I]

        print('Evaluating information criteria of the fit...')
        self.setInformation()

        print(self)
        return self


if __name__ == '__main__':
    print('Self test...')
    x1 = np.concatenate((np.random.randn(500,1),np.random.randn(500,1)+5),axis=0)
    x2 = np.concatenate((np.random.randn(500,1),np.random.randn(500,1)+10),axis=0)
    X = np.concatenate((x1,x2),axis=1)
    print('1000 x 2 matrix of X values produced:')
    print('X[:,1] is 50% mu=0')
    print('          50% mu=5')
    print('X[:,2] is 50% mu=0')
    print('          50% mu=10')
    print('Overall means: ')
    print(np.nanmean(X,axis=0))
    print('\n\n')

    print('Underfitting a null, 1-component model.')
    null = GMobj()
    null = null.fit(X,1,nrep=1,maxiter=10)

    print('Fitting a 2-component mixture model.')
    mixture = GMobj()
    mixture = mixture.fit(X,2)

    print('Overfitting a 3-component mixture model.')
    over = GMobj()
    over = over.fit(X,3)

    mixture.plot2d()
    plt.show()