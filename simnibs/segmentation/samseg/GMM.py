#import numpy as np
import numpy as npy
import torch as np

from scipy.stats import invwishart

eps = np.finfo( float ).eps

def expand_dims(x, dim):
    return x.unsqueeze(dim)


class GMM:
    def __init__(self, numberOfGaussiansPerClass, numberOfContrasts, useDiagonalCovarianceMatrices=True,
                 initialMeans=None, initialVariances=None,
                 initialMixtureWeights=None,
                 initialHyperMeans=None, initialHyperMeansNumberOfMeasurements=None, initialHyperVariances=None,
                 initialHyperVariancesNumberOfMeasurements=None, initialHyperMixtureWeights=None,
                 initialHyperMixtureWeightsNumberOfMeasurements=None ):
        #
        self.numberOfGaussiansPerClass = np.tensor(numberOfGaussiansPerClass).cuda()
        self.numberOfClasses = len(self.numberOfGaussiansPerClass)
        self.numberOfGaussians = sum(self.numberOfGaussiansPerClass)
        self.numberOfContrasts = numberOfContrasts
        self.useDiagonalCovarianceMatrices = useDiagonalCovarianceMatrices

        self.means = initialMeans
        self.variances = initialVariances
        self.mixtureWeights = initialMixtureWeights

        self.tied = False
        self.gaussNumber1Tied = None
        self.gaussNumber2Tied = None
        self.rho = None
        self.previousVariances = None

        print("AA")

        # Define the hyperparameters
        if initialHyperMeans is None:
            self.hyperMeans = np.zeros((self.numberOfGaussians, self.numberOfContrasts)).cuda()
        else:
            self.hyperMeans = initialHyperMeans.cuda()
        if initialHyperMeansNumberOfMeasurements is None:
            self.fullHyperMeansNumberOfMeasurements = np.zeros(self.numberOfGaussians).cuda()
        else:
            self.fullHyperMeansNumberOfMeasurements = initialHyperMeansNumberOfMeasurements.clone().cuda()
        if initialHyperVariances is None:
            self.hyperVariances = np.tile(np.eye(self.numberOfContrasts), (self.numberOfGaussians, 1, 1)).cuda()
        else:
            self.hyperVariances = initialHyperVariances
        if initialHyperVariancesNumberOfMeasurements is None:
            self.fullHyperVariancesNumberOfMeasurements = np.zeros(self.numberOfGaussians).cuda()
        else:
            self.fullHyperVariancesNumberOfMeasurements = initialHyperVariancesNumberOfMeasurements.clone().cuda()
        if initialHyperMixtureWeights is None:
            self.hyperMixtureWeights = np.ones(self.numberOfGaussians).cuda()
            for classNumber in range(self.numberOfClasses):
                # mixture weights are normalized (those belonging to one mixture sum to one)
                numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
                gaussianNumbers = (np.sum(self.numberOfGaussiansPerClass[:classNumber]) + np.arange(numberOfComponents).cuda()).to(np.int64)
                self.hyperMixtureWeights[gaussianNumbers] /= np.sum(self.hyperMixtureWeights[gaussianNumbers])
        else:
            self.hyperMixtureWeights = initialHyperMixtureWeights
        if initialHyperMixtureWeightsNumberOfMeasurements is None:
            self.fullHyperMixtureWeightsNumberOfMeasurements = np.zeros(self.numberOfClasses).cuda()
        else:
            self.fullHyperMixtureWeightsNumberOfMeasurements = initialHyperMixtureWeightsNumberOfMeasurements.clone().cuda()

        # Making sure the inverse-Wishart is normalizable (flat or peaked around hyperVariances)
        # requires that hyperVarianceNumberOfMeasurements is not smaller than (numberOfContrasts-1)
        # for any Gaussian. However, in order to prevent numerical errors with near-zero variances
        # (which can happen when almost no voxels are associated with a Gaussian in the EM algorithm,
        # due to e.g., tiny mixture weight), we use (numberOfContrasts-2)+1 instead.
        threshold = (self.numberOfContrasts - 2) + 1 + eps
        self.fullHyperVariancesNumberOfMeasurements[self.fullHyperVariancesNumberOfMeasurements < threshold] = threshold

        self.hyperMeansNumberOfMeasurements = self.fullHyperMeansNumberOfMeasurements.clone()
        self.hyperVariancesNumberOfMeasurements = self.fullHyperVariancesNumberOfMeasurements.clone()
        self.hyperMixtureWeightsNumberOfMeasurements = self.fullHyperMixtureWeightsNumberOfMeasurements.clone()

    def initializeGMMParameters(self, data, classPriors):

        print(data.dtype)
        print(classPriors.dtype)

        # Initialize the mixture parameters
        print("Initialize the mixture parameters")
        self.means = np.zeros((self.numberOfGaussians, self.numberOfContrasts)).cuda()
        self.variances = np.zeros((self.numberOfGaussians, self.numberOfContrasts, self.numberOfContrasts)).cuda()
        self.mixtureWeights = np.zeros(self.numberOfGaussians).cuda()
        for classNumber in range(self.numberOfClasses):
            # Calculate the global weighted mean and variance of this class, where the weights are given by the prior
            #print("Calculate the global weighted mean and variance of this class")
            prior = classPriors[:, classNumber]
            mean = data.T @ prior / np.sum(prior)
            tmp = data - mean
            prior = prior.unsqueeze(1)
            variance = tmp.T @ (tmp * prior) / np.sum(prior)
            if self.useDiagonalCovarianceMatrices:
                # Force diagonal covariance matrices
                variance = np.diag(np.diag(variance))

            # Based on this, initialize the mean and variance of the individual Gaussian components in this class'
            # mixture model: variances are simply copied from the global class variance, whereas the means are
            # determined by splitting the [ mean-sqrt( variance ) mean+sqrt( variance ) ] domain into equal intervals,
            # the middle of which are taken to be the means of the Gaussians. Mixture weights are initialized to be
            # all equal.

            # This actually creates a mixture model that mimics the single Gaussian quite OK-ish
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]

            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[: classNumber]) + componentNumber
                self.variances[gaussianNumber, :, :] = variance
                intervalSize = 2 * np.sqrt(np.diag(variance)) / numberOfComponents
                self.means[gaussianNumber, :] = (mean - np.sqrt(np.diag(variance)) + intervalSize / 2 +
                                            componentNumber * intervalSize).T
                self.mixtureWeights[gaussianNumber] = 1 / numberOfComponents

        if self.tied:
            self.previousVariances = self.variances.clone()

    def getGaussianLikelihoods(self, data, mean, variance):

        #
        L = np.linalg.cholesky(variance)
        tmp = np.linalg.solve_triangular(L, data.T - mean, upper=False)
        squaredMahalanobisDistances = np.sum(tmp ** 2, axis=0)
        sqrtDeterminantOfVariance = np.prod(np.diag(L))
        scaling = 1.0 / (2 * np.pi) ** (self.numberOfContrasts / 2) / sqrtDeterminantOfVariance
        gaussianLikelihoods = np.exp(squaredMahalanobisDistances * -0.5) * scaling
        gaussianLikelihoods = gaussianLikelihoods.cuda()
        return gaussianLikelihoods.T

    def getGaussianPosteriors(self, data, classPriors):

        #
        numberOfVoxels = data.shape[0]

        #gaussianPosteriors = np.zeros((numberOfVoxels, self.numberOfGaussians), order='F')
        gaussianPosteriors = np.zeros((numberOfVoxels, self.numberOfGaussians)).cuda()
        for classNumber in range(self.numberOfClasses):
            classPrior = classPriors[:, classNumber]
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mean = self.means[gaussianNumber, :].unsqueeze(1)
                variance = self.variances[gaussianNumber, :, :]

                gaussianLikelihoods = self.getGaussianLikelihoods(data, mean, variance)
                gaussianPosteriors[:, gaussianNumber] = gaussianLikelihoods * (
                            self.mixtureWeights[gaussianNumber] * classPrior)
        normalizer = np.sum(gaussianPosteriors, axis=1) + eps
        gaussianPosteriors = gaussianPosteriors / normalizer.unsqueeze(1)

        minLogLikelihood = -np.sum(np.log(normalizer))

        return gaussianPosteriors, minLogLikelihood

    def getLikelihoods(self, data, fractionsTable):
        #
        numberOfVoxels = data.shape[0]
        numberOfStructures = fractionsTable.shape[1]

        #
        likelihoods = np.zeros((numberOfVoxels, numberOfStructures), dtype=np.float64).cuda()
        for classNumber in range(self.numberOfClasses):

            # Compute likelihood for this class
            classLikelihoods = np.zeros(numberOfVoxels).cuda()
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mean = self.means[gaussianNumber, :].unsqueeze(1)
                variance = self.variances[gaussianNumber, :, :]
                mixtureWeight = self.mixtureWeights[gaussianNumber]

                gaussianLikelihoods = self.getGaussianLikelihoods(data, mean, variance)
                classLikelihoods += gaussianLikelihoods * mixtureWeight

            # Add contribution to the actual structures
            for structureNumber in range(numberOfStructures):
                fraction = fractionsTable[classNumber, structureNumber]
                if fraction < 1e-10:
                    continue
                likelihoods[:, structureNumber] += classLikelihoods * fraction

        #
        return likelihoods
    

    def getPosteriors(self, data, priors, fractionsTable):

        # Weight likelihood against prior and normalize
        posteriors = self.getLikelihoods(data, fractionsTable) * priors
        normalizer = np.sum(posteriors, axis=1) + eps
        posteriors = posteriors / normalizer.unsqueeze(1)

        return posteriors

    def fitGMMParameters(self, data, gaussianPosteriors):

        # Means and variances
        hyperMean = self.hyperMeans.unsqueeze(2)
        #hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements.unsqueeze(1).unsqueeze(1)
        hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements.unsqueeze(0)
        #print(self.hyperMeans.shape)
        #print(hyperMeanNumberOfMeasurements.shape)
        hyperMeanSum = hyperMean * hyperMeanNumberOfMeasurements
        hyperMeanProd =  1 \
                / (np.sum(gaussianPosteriors.reshape(-1,self.numberOfGaussians),dim=0,keepdim=False).unsqueeze(0) + hyperMeanNumberOfMeasurements)

        #print(data.shape)
        means = (data.T @ gaussianPosteriors.reshape(-1,self.numberOfGaussians) + self.hyperMeans.T) * hyperMeanProd
        #print(means.shape)
        #print(hyperMeanProd.shape)
        hyperVariance = self.hyperVariances
        hyperVarianceNumberOfMeasurements = self.hyperVariancesNumberOfMeasurements.unsqueeze(1).unsqueeze(1)
        hyperVarianceProd = 1 \
                / (np.sum(gaussianPosteriors.reshape(-1,self.numberOfGaussians),dim=0,keepdim=False).unsqueeze(1).unsqueeze(1) + hyperVarianceNumberOfMeasurements)
        hyperVarianceSum = hyperVariance * hyperVarianceNumberOfMeasurements
        #print(hyperVarianceProd.shape)
        #print(hyperVarianceSum.shape)
        for gaussianNumber in range(self.numberOfGaussians):
            posterior = gaussianPosteriors[:, gaussianNumber].reshape(-1, 1)

            mean = means[:,gaussianNumber].unsqueeze(1)
            #print(mean.shape)
            tmp = data - mean.T
            variance = (tmp.T @ (tmp * posterior) + \
                        hyperMeanNumberOfMeasurements[0,gaussianNumber] * ((mean - hyperMean[gaussianNumber]) @ (mean - hyperMean[gaussianNumber]).T) + hyperVarianceSum[gaussianNumber])\
                        * hyperVarianceProd[gaussianNumber]
            if self.useDiagonalCovarianceMatrices:
                # Force diagonal covariance matrices
                variance = np.diag(np.diag(variance))
            self.variances[gaussianNumber, :, :] = variance
            self.means[gaussianNumber, :] = mean.T

        # Mixture weights
        #print("Mixture weights")
        self.mixtureWeights = np.sum(gaussianPosteriors + eps, axis=0)
        for classNumber in range(self.numberOfClasses):
            # mixture weights are normalized (those belonging to one mixture sum to one)
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            gaussianNumbers = (np.sum(self.numberOfGaussiansPerClass[:classNumber]) + np.arange(numberOfComponents).cuda()).to(np.int64)

            self.mixtureWeights[gaussianNumbers] += self.hyperMixtureWeights[gaussianNumbers] * \
                                               self.hyperMixtureWeightsNumberOfMeasurements[classNumber]
            self.mixtureWeights[gaussianNumbers] /= np.sum(self.mixtureWeights[gaussianNumbers])

        if self.tied:
            self.tiedGaussiansFit(data, gaussianPosteriors)

    def evaluateMinLogPriorOfGMMParameters(self):
        #
        minLogPrior = 0
        Ls = np.linalg.cholesky(self.variances)  # variance = L @ L.T
        hyperLs = np.linalg.cholesky(self.hyperVariances)  # hyperVariance = hyperL @ hyperL.T
        for gaussianNumber in range(self.numberOfGaussians):
            mean = self.means[gaussianNumber, :].unsqueeze(1)
            variance = self.variances[gaussianNumber, :, :]

            hyperMean = expand_dims(self.hyperMeans[gaussianNumber, :], 1)
            hyperMeanNumberOfMeasurements = self.hyperMeansNumberOfMeasurements[gaussianNumber]
            hyperVariance = self.hyperVariances[gaussianNumber, :, :]
            hyperVarianceNumberOfMeasurements = self.hyperVariancesNumberOfMeasurements[gaussianNumber]

            # -log N( mean | hyperMean, variance / hyperMeanNumberOfMeasurements )
            #L = np.linalg.cholesky(variance)  # variance = L @ L.T
            L = Ls[gaussianNumber]
            halfOfLogDetVariance = np.sum(np.log(np.diag(L)))
            tmp = np.linalg.solve_triangular(L, mean - hyperMean, upper=False)
            squaredMahalanobisDistance = np.sum(tmp * tmp)
            minLogPrior += squaredMahalanobisDistance * hyperMeanNumberOfMeasurements / 2 + halfOfLogDetVariance

            # -log IW( variance | hyperVariance * hyperVarianceNumberOfMeasurements,
            #                     hyperVarianceNumberOfMeasurements - numberOfContrasts - 2 )
            #
            #hyperL = np.linalg.cholesky(hyperVariance)  # hyperVariance = hyperL @ hyperL.T
            hyperL = hyperLs[gaussianNumber]
            halfOfLogDetHyperVariance = np.sum(np.log(np.diag(hyperL)))
            tmp = np.linalg.solve_triangular(L, hyperL, upper=False)
            minLogPrior += np.trace(tmp @ tmp.T) * hyperVarianceNumberOfMeasurements / 2 + \
                           hyperVarianceNumberOfMeasurements * halfOfLogDetVariance - \
                           (hyperVarianceNumberOfMeasurements - self.numberOfContrasts - 2) * halfOfLogDetHyperVariance

        for classNumber in range(self.numberOfClasses):
            # -log Dir( weights | hyperMixtureWeights * hyperMixtureWeightNumberOfMeasurements + 1 )
            hyperMixtureWeightNumberOfMeasurements = self.hyperMixtureWeightsNumberOfMeasurements[classNumber]
            numberOfComponents = self.numberOfGaussiansPerClass[classNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = sum(self.numberOfGaussiansPerClass[:classNumber]) + componentNumber
                mixtureWeight = self.mixtureWeights[gaussianNumber]
                hyperMixtureWeight = self.hyperMixtureWeights[gaussianNumber]

                # minLogPrior -= hyperMixtureWeight * hyperMixtureWeightNumberOfMeasurements * np.log( mixtureWeight )
                #
                # I'm using Stirling's approximation on the normalizing constant (beta function) just in the same way
                # as in Appendix C of Van Leemput TMI 2009
                minLogPrior += hyperMixtureWeightNumberOfMeasurements * \
                               hyperMixtureWeight * (np.log(hyperMixtureWeight + eps) - np.log(mixtureWeight + eps))

        #
        return minLogPrior

    def downsampledHyperparameters(self, downSamplingFactors):
        self.hyperMeansNumberOfMeasurements = self.fullHyperMeansNumberOfMeasurements / np.prod(downSamplingFactors)
        self.hyperVariancesNumberOfMeasurements = self.fullHyperVariancesNumberOfMeasurements / np.prod(downSamplingFactors)
        self.hyperMixtureWeightsNumberOfMeasurements = self.fullHyperMixtureWeightsNumberOfMeasurements / np.prod(downSamplingFactors)

    # Tied gaussian 2 to gaussian 1 (rho indicates outlier factor measured in std deviations)
    # Note that this works only if the classes have only one single component
    def tiedGaussiansInit(self, gaussNumber1, gaussNumber2, rho):
        self.tied = True
        self.gaussNumber1Tied = gaussNumber1
        self.gaussNumber2Tied = gaussNumber2
        self.rho = rho

    def tiedGaussiansFit(self, data, gaussianPosteriors):

        posterior_1 = gaussianPosteriors[:, self.gaussNumber1Tied].reshape(-1, 1)
        hyperMean_1 = expand_dims(self.hyperMeans[self.gaussNumber1Tied, :], 1)
        hyperMeanNumberOfMeasurements_1 = self.hyperMeansNumberOfMeasurements[self.gaussNumber1Tied]
        hyperVariance_1 = self.hyperVariances[self.gaussNumber1Tied, :, :]
        hyperVarianceNumberOfMeasurements_1 = self.hyperVariancesNumberOfMeasurements[self.gaussNumber1Tied]
        variance_1_previous = self.previousVariances[self.gaussNumber1Tied]

        posterior_2 = gaussianPosteriors[:, self.gaussNumber2Tied].reshape(-1, 1)
        hyperMean_2 = expand_dims(self.hyperMeans[self.gaussNumber2Tied, :], 1)
        hyperMeanNumberOfMeasurements_2 = self.hyperMeansNumberOfMeasurements[self.gaussNumber2Tied]
        hyperVariance_2 = self.hyperVariances[self.gaussNumber2Tied, :, :]
        hyperVarianceNumberOfMeasurements_2 = self.hyperVariancesNumberOfMeasurements[self.gaussNumber2Tied]
        variance_2_previous = self.previousVariances[self.gaussNumber2Tied]

        # Define some temporary variables
        soft_sum_1 = np.sum(posterior_1)
        soft_sum_2 = np.sum(posterior_2)
        g12TiedMean = ((hyperMeanNumberOfMeasurements_2 * soft_sum_2) /
                         (soft_sum_2 + hyperMeanNumberOfMeasurements_2)) * \
                        np.linalg.solve(variance_2_previous, variance_1_previous)

        # Updates for the means
        mean_1 = np.linalg.solve((soft_sum_1 + hyperMeanNumberOfMeasurements_1) * np.eye(self.numberOfContrasts) +
                                 g12TiedMean, (data.T @ posterior_1 + hyperMean_1 * hyperMeanNumberOfMeasurements_1 +
                                               g12TiedMean @ ((data.T @ posterior_2) / soft_sum_2)))

        mean_2 = (data.T @ posterior_2 + mean_1 * hyperMeanNumberOfMeasurements_2) / \
                 (soft_sum_2 + hyperMeanNumberOfMeasurements_2)

        # Updates for the variances
        tmp = data - mean_1.T

        g12TiedVariance = (2 + self.numberOfContrasts) * np.eye(self.numberOfContrasts) +\
                          hyperVarianceNumberOfMeasurements_2 * (np.linalg.solve(variance_2_previous, self.rho * variance_1_previous)
                                                                 - np.eye(self.numberOfContrasts))

        variance_1 = np.linalg.solve((soft_sum_1 + hyperVarianceNumberOfMeasurements_1) *
                                     np.eye(self.numberOfContrasts) + g12TiedVariance,
                                     (tmp.T @ (tmp * posterior_1) + hyperMeanNumberOfMeasurements_1 *
                                      ((mean_1 - hyperMean_1) @ (mean_1 - hyperMean_1).T) + hyperVariance_1))

        tmp = data - mean_2.T

        variance_2 = (tmp.T @ (tmp * posterior_2) + hyperMeanNumberOfMeasurements_2 *
                      ((mean_2 - mean_1) @ (mean_2 - mean_1).T) +
                      hyperVarianceNumberOfMeasurements_2 * self.rho * variance_1) /\
                     (soft_sum_2 + hyperVarianceNumberOfMeasurements_2)

        if self.useDiagonalCovarianceMatrices:
            variance_2 = np.diag(np.diag(variance_2))
            variance_1 = np.diag(np.diag(variance_1))

        self.means[self.gaussNumber1Tied, :] = mean_1.T
        self.means[self.gaussNumber2Tied, :] = mean_2.T
        self.variances[self.gaussNumber1Tied, :, :] = variance_1
        self.variances[self.gaussNumber2Tied, :, :] = variance_2

        self.previousVariances[self.gaussNumber1Tied, :, :] = variance_1
        self.previousVariances[self.gaussNumber2Tied, :, :] = variance_2

        self.hyperMeans[self.gaussNumber2Tied] = self.means[self.gaussNumber1Tied]
        self.hyperVariances[self.gaussNumber2Tied] = self.rho * self.variances[self.gaussNumber1Tied]

    def sampleMeansAndVariancesConditioned(self, data, posterior, gaussianNumber):
        tmpGmm = GMM([1], self.numberOfContrasts, self.useDiagonalCovarianceMatrices,
                  initialHyperMeans=np.tensor([self.hyperMeans[gaussianNumber]]).cuda(),
                  initialHyperMeansNumberOfMeasurements=np.tensor([self.hyperMeansNumberOfMeasurements[gaussianNumber]]).cuda(),
                  initialHyperVariances=np.tensor([self.hyperVariances[gaussianNumber]]).cuda(),
                  initialHyperVariancesNumberOfMeasurements=np.tensor([self.hyperVariancesNumberOfMeasurements[gaussianNumber]])).cuda()
        tmpGmm.initializeGMMParameters(data, posterior)
        tmpGmm.fitGMMParameters(data, posterior)
        N = posterior.sum()

        # Murphy, page 134 with v0 = hyperVarianceNumberOfMeasurements - numberOfContrasts - 2
        variance = invwishart.rvs(N + tmpGmm.hyperVariancesNumberOfMeasurements[0] - self.numberOfContrasts - 2,
                                  tmpGmm.variances[0] * (tmpGmm.hyperVariancesNumberOfMeasurements[0] + N))

        # If numberOfContrast is 1 force variance to be a (1,1) array
        if self.numberOfContrasts == 1:
            variance = np.atleast_2d(variance)

        if self.useDiagonalCovarianceMatrices:
            variance = np.diag(np.diag(variance))

        mean = np.random.multivariate_normal(tmpGmm.means[0],
                                             variance / (tmpGmm.hyperMeansNumberOfMeasurements[0] + N)).reshape(-1, 1)
        return mean, variance
    
