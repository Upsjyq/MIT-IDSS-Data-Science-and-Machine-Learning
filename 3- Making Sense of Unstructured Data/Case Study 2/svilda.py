import random
import numpy as n
from scipy.special import psi
from nltk.tokenize import wordpunct_tokenize

meanchangethresh = 1e-3

def dirichlet_expectation(alpha):
    '''see onlineldavb.py by Blei et al'''
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])

class SVILDA():
    def __init__(self, vocab, K, D, alpha, eta, tau, kappa, iterations):
        self._vocab = vocab
        self._V = len(vocab)
        self._K = K
        self._D = D
        self._alpha = alpha
        self._eta = eta
        self._tau = tau+1
        self._kappa = kappa
        self._lambda = 1* n.random.gamma(100., 1./100., (self._K, self._V))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self.ct = 0
        self._iterations = iterations

    def updateLocal(self, doc): #word_dn is an indicator variable with dimension V
        (wordids, wordcts) = doc
        batchD = len(wordids)

        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch
        gamma = 1*n.random.gamma(100., 1./100., (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)
        # Now, for each document d update that document's gamma and phi
        it = 0
        meanchange = 0
        for d in range(0, batchD):
            # These are mostly just shorthand (but might help cache locality)
            ids = wordids[d]
            cts = wordcts[d]
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self._expElogbeta[:, ids]
            # The optimal phi_{dwk} is proportional to 
            # expElogthetad_k * expElogbetad_w. phinorm is the normalizer.
            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
            # Iterate between gamma and phi until convergence
            for it in range(0, 100):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self._alpha + n.multiply(expElogthetad, expElogbetad) * cts / phinorm
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100
                # If gamma hasn't changed much, we're done.
                meanchange = n.mean(abs(gammad - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            gamma[d, :] = gammad
            # Contribution of document d to the expected sufficient
            # statistics for the M step.
            sstats[:, ids] += n.multiply(expElogthetad, expElogbetad) * cts / phinorm

        # This step finishes computing the sufficient statistics for the
        # M step, so that
        # sstats[k, w] = \sum_d n_{dw} * phi_{dwk} 
        # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.

        return gamma, sstats

    def updateGlobal(self, sstats, doc):
            # print 'updating global parameters'
        (words, counts) = doc
            
        rhot = (self.ct + self._tau) **(-self._kappa)

        self._lambda = self._lambda * (1-rhot) + rhot * (self._eta + self._D * sstats / len(words))
            
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    
    def runSVI(self, doc_vecs):

        for i in range(self._iterations):            
            randint = random.randint(0, self._D-1)
            if (i%100)==0: 
                print("ITERATION", i, " running document number ", randint)
            
            doc = doc_vecs[randint]
            gamma_d, sstats = self.updateLocal(doc)
            self.updateGlobal(sstats, doc)
            self.ct += 1
