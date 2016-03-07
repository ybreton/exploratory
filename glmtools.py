import math
import numpy as np
import matplotlib.pyplot as plt

def pca(mat,pairwise=False):
    """
    Conduct a principal components analysis of the n x d matrix mat,
    returning a list of d x nEigs tuples with normalized eigenvector (pc component) and normalized eigenvalue (PC weight)

    :param mat: n x d matrix of n observations of d variables
    :param pairwise (optional bool, default False): exclude missing or inf values pairwise in covariance calculation, or entire row
    :return: PCout      is a 1 x nEigs list of the following tuple:
            (Normalized Eigenvalue, [1xd Normalized Eigenvector])
            for each principal component in the list.
            The sum of all normalized eigenvalues is 1; the normalized eigenvector has unit length.

    """
    mat = np.array(mat)
    m = np.nanmean(mat,axis=0)
    # mean of inputs
    D = mat-m
    # deviations of inputs from mean (residuals)
    idInvalid = np.array(np.logical_or(np.isnan(D),np.isinf(D)))
    print('Calculating variance-covariance matrix...')
    if pairwise:
        (n0,d) = np.shape(D)
        cov = np.zeros((d,d))
        for iC1 in range(d-1):
            for iC2 in range(iC1,d):
                idPair = np.logical_or(idInvalid[:,iC1],idInvalid[:,iC2])
                idValid = np.logical_not(idPair)
                x = D[:,iC1][idValid]
                y = D[:,iC2][idValid]
                n = np.sum(idValid)
                if n==0:
                    print(str(iC1),',',str(iC2),' have no observations.')
                else:
                    cov[iC1,iC2] = np.dot(x.T,y)/n
    else:
        idValid = np.any(idInvalid,axis=1)
        n = np.sum(np.logical_not(idInvalid),axis=0)

        D = D[idValid,:]
        cov = np.dot(D.T,D)/n

    # covariance of x,y is mean product of deviation of x with deviation of y
    print('Extracting eigenvalues and eigenvectors...')
    w, v = np.linalg.eig(cov)
    # get eigenvalues and eigenvectors
    eps = 1*10**-32
    t = np.sum(w)+eps
    p = [(w[ipc]+eps)/t for ipc in range(len(w))]
    # proportion of total variance explained
    pcs = np.argsort(-np.array(p))

    outlist = list()
    for iPC in pcs:
        t = (p[iPC], v[:,iPC])
        outlist.append(t)

    return outlist


def evalpca(X, PCA, PCs=None, PCprop=None):
    """
    Evaluate the principal components as returned by pca2.

    :param X:  n x d array of input predictors
    :param PCA: 1 x nEigs list of (Normalized Eigenvalue, [1xd Normalized Eigenvector]) tuples
    Optional parameter keywords
    :param PCs: Number of principal components to evaluate
    :param PCprop: Minimum total proportion of variance to account for

    :return: n x PCs array of principal component scores for each case n on each component PCs.

    """
    if PCs is None:
        if PCprop is None:
            # If the number of PCs is not set, do all in the list
            PCs = len(PCA)
        else:
            # If the number of PCs is not set, but we want a minimum variance accounted for
            cum = list()
            (cum[0],eig[0]) = PCA[0]
            for ir in range(1,len(PCA)):
                p,v = PCA[ir]
                cum.append(cum[ir-1]+p)
                print('Features 1:',str(ir+1),' cum variance - ',str(cum[ir]))
            gt = [ix>=PCprop for ix in cum]
            pcs = np.arange(len(cum))
            pcgt= pcs[np.array(gt)]
            PCs = min(pcgt)

    eig = [PCA[ir][1] for ir in range(len(PCA))]
    print('Producing scores for',str(PCs+1),'features')

    n = X.shape[0]
    D = X-np.nanmean(X,axis=0)
    Fout = np.ones((n,PCs))*np.nan
    for iPC in range(PCs):
        print('Evaluating PC',iPC)
        Fout[:,iPC] = np.dot(D,eig[iPC].T).T

    return Fout

def eigenplot(PCA):
    """
    Plots the magnitudes of the normalized eigenvalues in PCA, and the cumulative magnitudes.

    :param PCA: 1 x nEigs list of principal component tuples (Normalized eigenvalue, [d-dimensional Normalized eigenvector])
    :return: (fh, ah, ph) tuple of handles to figure object

    """

    y = [PCA[i][0] for i in range(len(PCA))]
    cy = list(y[:])
    for i in range(1,len(y)):
        cy[i] = cy[i-1]+y[i]
    x = np.arange(1,len(PCA)+1)
    fh = plt.figure()
    fh.suptitle('Normalized eigenvalue plot')
    ah1 = fh.add_subplot(2,1,1)
    ph1 = plt.plot(x,y,'ko-')
    ah1.set_xlabel('Principal Component')
    ah1.set_ylabel('Normalized eigenvalue')
    ah1.set_ylim(0,y[0])
    ah1.set_xlim(0,len(PCA)+1)
    ah2 = fh.add_subplot(2,1,2)
    ph2 = plt.plot(x,cy,'ko-')
    ah2.set_xlabel('Principal Component')
    ah2.set_ylabel('Cumulative variance accounted for')
    ah2.set_ylim(0,1)
    ah2.set_xlim(0,len(PCA)+1)

    ah = (ah1,ah2)
    ph = (ph1,ph2)
    return (fh, ah, ph)

def biplot(X, PCA):
    """
    Plots a PCA biplot of PC scores for each observation and loadings from each variable.

    :param X: n x d matrix of n observations of d variables
    :param PCA: 1 x nEigs list of tuples (Normalized eigenvalue, [Normalized Eigenvectors])
    :return: (fh,ah,ph) tuple of figure handles, where ah is a list of all axis handles and ph is a list of tuples of the plot handles.

    """

    X = np.array(X)
    X0 = evalpca2(X, PCA)
    try:
        (n,d) = X0.shape
    except:
        n = X0.shape[0]
        d = 1


    k = len(PCA)

    eigenvalues = [PCA[i][0] for i in range(len(PCA))]
    eigenvectors = [PCA[i][1] for i in range(len(PCA))]
    fh = plt.figure()

    ahList = []
    phList = []

    if k>2:
        for iC1 in range(k-1):
            for iC2 in range(iC1+1,k):
                p = (iC2*k)+(iC1+1)
                f1 = X0[:,iC1]
                f2 = X0[:,iC2]


                l1 = eigenvectors[iC1]
                l2 = eigenvectors[iC2]

                print('Feature',str(iC1),'vs','Feature',str(iC2))
                ah=fh.add_subplot(k+1,k+1,p)
                ph1=plt.plot(f1,f2,'ko',markerfacecolor='k',markersize=5)
                ph2=plt.plot(l1,l2,'rs',markerfacecolor='r',markersize=12)
                for iN in range(d):
                    plt.text(l1[iN],l2[iN],str(iN))
                xh=plt.xlabel('F'+str(iC1))
                xh.set_fontsize(10)
                yh=plt.ylabel('F'+str(iC2))
                yh.set_fontsize(10)
                fstr = '{0:.3f}+{1:.3f}'.format(eigenvalues[iC1],eigenvalues[iC2])
                th=plt.title(fstr)
                th.set_fontsize(10)
                ph = (ph1,ph2)
                phList.append(ph)
                ahList.append(ah)
    elif k==2:
        iC1 = 0
        iC2 = 1
        f1 = X0[:,iC1]
        f2 = X0[:,iC2]

        l1 = PClist[:,iC1]
        l2 = PClist[:,iC2]

        print('Feature',str(iC1),'vs','Feature',str(iC2))
        ah=fh.add_subplot(1,1,1)
        ph1=plt.plot(f1,f2,'ko',markerfacecolor='k',markersize=5)
        ph2=plt.plot(l1,l2,'rs',markerfacecolor='r',markersize=12)
        th=plt.text(l1,l2,np.arange(d))
        plt.xlabel('F'+str(iC1))
        plt.ylabel('F'+str(iC2))
        plt.title(str(PCweight[iC1])+'+ '+str(PCweight[iC2]))
        ph = (ph1,ph2)
        phList.append(ph)
        ahList.append(ah)
    else:
        ah=fh.add_subplot(1,1,1)
        ph1=plt.plot(np.ones(X0.shape),X0,'ko',markerfacecolor='k',markersize=5)
        ph2=plt.plot(np.ones(X0.shape),PClist,'rs',markerfacecolor='r',markersize=12)
        ph = (ph1,ph2)
        phList = [ph]
        ahList = [ah]

    return (fh,ahList,phList)



def makeOrdinal(x,emptyCat=True,return_counts=False, return_labels=False):

    """
    Turns the categorical variable in x into an ordinal variable according to its relative frequency in x.
    Missing values can also get their own category, specially marked as 0.

    :param x: n-list of categorical values, from 1 through the number of k unique category labels in x
    :param emptyCat=True, bool, optional
        if true, missing values are assigned category 0
    :param return_counts=False, bool, optional
        if true, returns k-list of category counts
    :param return_labels=False, bool, optional
        if true, returns k-list of category labels
    :return: o: n-list of ordinal values based on categorical frequency count, from highest to lowest; 0 is missing value category
            (o, counts)
            (o, labels)
            (o, counts, labels)

    """
    x = np.array(x)
    idExc = []
    for ix in x:
        if ix is None:
            idExc.append(True)
        elif isinstance(ix,str):
            I = ix.lower=='nan' or ix.lower=='na'
            idExc.append(I)
        else:
            I = np.isnan(ix)
            idExc.append(I)
    idExc = np.array(idExc,dtype=bool)
    u = np.unique(x[np.logical_not(idExc)])

    c = [np.nansum(x==iu) for iu in u]
    i = np.argsort(c)
    i = i[::-1]
    U = u[i]
    o = np.ones((x.shape), dtype=int)*np.nan
    k = 1
    for iU in U:
        idx = x == iU
        o[idx] = k
        k += 1

    counts = []
    labels = ['nan']
    if emptyCat:
        o[idExc] = 0
        counts.append(np.nansum(idExc))
    else:
        counts.append(np.nan)
    counts[1:] = c
    labels[1:] = U

    o = list(o)

    if return_counts and not return_labels:
        return (o,counts)
    elif return_counts and return_labels:
        return (o,counts,labels)
    elif not return_counts and return_labels:
        return (o,labels)
    else:
        return o


def crosscorrs(mat,plotFlag=False):
    """
    Produces the cross-correlation matrix of the columns of mat, and optionally plots the cross-correlation matrix

    :param mat: n x d matrix of n observations of d variables
    :param plotFlag=False, bool: true/false flag for plotting the cross-correlation matrix.
    :return: d x d matrix of cross-correlations of d variables
    """

    try:
        (n,d) = mat.shape
    except:
        n = len(mat)
        d = 1
    R = np.ones((d,d))
    for iC1 in range(d-1):
        for iC2 in range(iC1+1, d):
            x = mat[:, iC1]
            y = mat[:, iC2]
            id1 = np.logical_or(np.isnan(x), np.isinf(x))
            id2 = np.logical_or(np.isnan(y), np.isinf(y))
            idValid = np.logical_and(np.logical_not(id1), np.logical_not(id2))
            x = x[idValid]
            y = y[idValid]
            r = np.corrcoef(x, y, rowvar=0)
            R[iC1,iC2] = r[0,1]
            R[iC2,iC1] = r[1,0]

    if plotFlag:
        fh,ah = plt.subplots()
        fh.suptitle('Cross-correlation matrix')
        ph = ah.imshow(R.T,vmin=-1,vmax=1,origin='lower')

        ah.set_xlabel('Column number')
        ah.set_ylabel('Column number')
        fh.colorbar(ph)
        handles = (fh,ah,ph)

        return (R,handles)
    else:
        return R


def histograms(mat):
    """
    Produces histograms of the data in each column of mat.

    :param mat: n x d matrix of n observations of d variables
    :return: fh: 1 x d list of handles to figures
    """
    try:
        (n,d) = mat.shape
    except:
        n = len(mat)
        d = 1

    fh = list()
    for iC1 in range(d):
        x = mat[:,iC1]
        idx = np.logical_or(np.isnan(x),np.isinf(x))
        idValid = np.logical_not(idx)
        x = x[idValid]
        n = len(x)
        nbin = np.ceil(math.sqrt(n))

        fh0 = plt.figure()
        fh0.suptitle('Distribution, column'+str(iC1+1))
        ah = fh0.add_subplot()
        ph = ah.hist(x,nbin)
        ah.set_xlim(np.nanmin(x),np.nanmax(x))
        fh.append(fh0)
    return fh


