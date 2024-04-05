#import numpy as np
import numpy as npy
import torch as np
import math

eps = np.finfo( float ).eps

def expand_dims(x, dim):
    return x.unsqueeze(dim)


class BiasField:
    def __init__(self, imageSize, smoothingKernelSize):
        self.fullBasisFunctions = self.getBiasFieldBasisFunctions(imageSize, smoothingKernelSize)
        self.basisFunctions = [f.cuda() for f in self.fullBasisFunctions.copy()]
        self.coefficients = None

    def backprojectKroneckerProductBasisFunctions(self, kroneckerProductBasisFunctions, coefficients):
        numberOfDimensions = len(kroneckerProductBasisFunctions)
        Ms = np.zeros(numberOfDimensions, dtype=np.int64).cuda()  # Number of basis functions in each dimension
        Ns = np.zeros(numberOfDimensions, dtype=np.int64).cuda()  # Number of basis functions in each dimension
        transposedKroneckerProductBasisFunctions = []
        for dimensionNumber in range(numberOfDimensions):
            Ms[dimensionNumber] = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
            Ns[dimensionNumber] = kroneckerProductBasisFunctions[dimensionNumber].shape[0]
            transposedKroneckerProductBasisFunctions.append(kroneckerProductBasisFunctions[dimensionNumber].T)
        y = self.projectKroneckerProductBasisFunctions(transposedKroneckerProductBasisFunctions,
                                                       coefficients.reshape(Ms.cpu().numpy().tolist()) )
        Y = y.reshape(Ns.cpu().numpy().tolist())
        return Y

    def projectKroneckerProductBasisFunctions(self, kroneckerProductBasisFunctions, T):
        #
        # Compute
        #   c = W' * t
        # where
        #   W = W{ numberOfDimensions } \kron W{ numberOfDimensions-1 } \kron ... W{ 1 }
        # and
        #   t = T( : )
        numberOfDimensions = len(kroneckerProductBasisFunctions)
        currentSizeOfT = list(T.shape)
        #print(T.device)
        for dimensionNumber in range(numberOfDimensions):
            # Reshape into 2-D, do the work in the first dimension, and shape into N-D
            T = T.reshape((currentSizeOfT[0], -1))
            #print(kroneckerProductBasisFunctions[dimensionNumber].device)
            T = ( kroneckerProductBasisFunctions[dimensionNumber] ).T @ T
            currentSizeOfT[0] = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
            T = T.reshape(currentSizeOfT)
            # Shift dimension
            currentSizeOfT = currentSizeOfT[1:] + [currentSizeOfT[0]]
            T = np.roll(T, 3, dims=0)
        # Return result as vector
        coefficients = T.flatten()
        return coefficients.contiguous()

    def computePrecisionOfKroneckerProductBasisFunctions(self, kroneckerProductBasisFunctions, B):
        #
        # Compute
        #   H = W' * diag( B ) * W
        # where
        #   W = W{ numberOfDimensions } \kron W{ numberOfDimensions-1 } \kron ... W{ 1 }
        # and B is a weight matrix
        numberOfDimensions = len( kroneckerProductBasisFunctions )

        # Compute a new set of basis functions (point-wise product of each combination of pairs) so that we can
        # easily compute a mangled version of the result
        Ms = np.zeros( numberOfDimensions, dtype=np.int64).cuda() # Number of basis functions in each dimension
        print("awa")
        hessianKroneckerProductBasisFunctions = {}
        for dimensionNumber in range(numberOfDimensions):
            M = kroneckerProductBasisFunctions[dimensionNumber].shape[1]
            A = kroneckerProductBasisFunctions[dimensionNumber].cuda()
            hessianKroneckerProductBasisFunctions[dimensionNumber] = np.kron( np.ones( (1, M )).cuda(), A ) * np.kron( A, np.ones( (1, M) ).cuda() ).cuda()
            print("awoo")
            Ms[dimensionNumber] = M
        result = self.projectKroneckerProductBasisFunctions( hessianKroneckerProductBasisFunctions, B ).cuda()
        print("awoowa")
        new_shape = list(np.kron( Ms, np.tensor([ 1, 1 ]).cuda() ))
        print("awa")
        new_shape.reverse()
        print("awa2")
        result = result.reshape(new_shape)
        print("awa3")
        permutationIndices = tuple(np.hstack((2 * np.arange(numberOfDimensions), 2 * np.arange(numberOfDimensions) +1)).numpy().tolist())
        print(permutationIndices)
        result = np.permute(result, permutationIndices)
        print("awa4")
        print(result.shape)
        new_shape = ( np.prod( Ms ), np.prod( Ms ) )
        print(new_shape)
        precisionMatrix = result.reshape(new_shape)
        print("mengaoo")
        return precisionMatrix.cuda()

    def getBiasFieldBasisFunctions(self, imageSize, smoothingKernelSize):
        # Our bias model is a linear combination of a set of basis functions. We are using so-called
        # "DCT-II" basis functions, i.e., the lowest few frequency components of the Discrete Cosine
        # Transform.
        biasFieldBasisFunctions = []
        for dimensionNumber in range(3):
            N = imageSize[dimensionNumber]
            delta = smoothingKernelSize[dimensionNumber]
            M = math.ceil(N / delta) + 1
            Nvirtual = (M - 1) * delta
            js = [(index + 0.5) * math.pi / Nvirtual for index in range(N)]
            scaling = [math.sqrt(2 / Nvirtual)] * M
            scaling[0] /= math.sqrt(2)
            A = np.tensor([[math.cos(freq * m) * scaling[m] for m in range(M)] for freq in js]).cuda()
            biasFieldBasisFunctions.append(A)

        return biasFieldBasisFunctions

    def getBiasFields(self, mask=None):
        #
        numberOfContrasts = self.coefficients.shape[-1]
        imageSize = tuple([functions.shape[0] for functions in self.basisFunctions])
        biasFields = np.zeros(imageSize + (numberOfContrasts,)).cuda()
        for contrastNumber in range(numberOfContrasts):
            biasField = self.backprojectKroneckerProductBasisFunctions(
                self.basisFunctions, self.coefficients[:, contrastNumber])
            if mask is not None:
                biasField *= mask
            biasFields[:, :, :, contrastNumber] = biasField

        return biasFields

    def fitBiasFieldParameters(self, imageBuffers, gaussianPosteriors, means, variances, mask):

        # Bias field correction: implements Eq. 8 in the paper
        #    Van Leemput, "Automated Model-based Bias Field Correction of MR Images of the Brain", IEEE TMI 1999

        #
        numberOfGaussians = means.shape[0]
        numberOfContrasts = means.shape[1]
        numberOfBasisFunctions = [functions.shape[1] for functions in self.basisFunctions]
        numberOf3DBasisFunctions = np.prod(np.tensor(numberOfBasisFunctions)).cuda()

        # Set up the linear system lhs * x = rhs
        precisions = np.zeros_like(variances).cuda()
        for gaussianNumber in range(numberOfGaussians):
            precisions[gaussianNumber, :, :] = np.linalg.inv(variances[gaussianNumber, :, :]).reshape(
                (1, numberOfContrasts, numberOfContrasts))

        lhs = np.zeros((numberOf3DBasisFunctions * numberOfContrasts,
                        numberOf3DBasisFunctions * numberOfContrasts)).cuda()  # left-hand side of linear system
        rhs = np.zeros((numberOf3DBasisFunctions * numberOfContrasts, 1)).cuda()  # right-hand side of linear system
        weightsImageBuffer = np.zeros(mask.shape).cuda()
        tmpImageBuffer = np.zeros(mask.shape).cuda()
        for contrastNumber1 in range(numberOfContrasts):
            # logger.debug('third time contrastNumber=%d', contrastNumber)
            contrast1Indices = np.arange(0, numberOf3DBasisFunctions).cuda() + \
                               contrastNumber1 * numberOf3DBasisFunctions

            tmp = np.zeros(gaussianPosteriors.shape[0]).cuda()
            for contrastNumber2 in range(numberOfContrasts):
                contrast2Indices = np.arange(0, numberOf3DBasisFunctions).cuda() + \
                                   contrastNumber2 * numberOf3DBasisFunctions

                classSpecificWeights = gaussianPosteriors * precisions[:, contrastNumber1, contrastNumber2]
                weights = np.sum(classSpecificWeights, 1)

                # Build up stuff needed for rhs
                predicted = np.sum(classSpecificWeights * means[:, contrastNumber2], 1) / (weights + eps)
                print(mask.shape)
                print(imageBuffers.shape)
                print(weights.shape)
                residue = imageBuffers[..., contrastNumber2][mask] - predicted
                tmp += weights * residue

                # Fill in submatrix of lhs
                weightsImageBuffer[mask] = weights
                print(contrast1Indices.device)
                print(contrast2Indices.device)
                lhs[contrast1Indices[:,None], contrast2Indices[None,:]] \
                    = self.computePrecisionOfKroneckerProductBasisFunctions(self.basisFunctions,
                                                                       weightsImageBuffer)

            print("aaaaaaaaaaanya")
            tmpImageBuffer[mask] = tmp
            rhs[contrast1Indices] = self.projectKroneckerProductBasisFunctions(self.basisFunctions,
                                                                          tmpImageBuffer).reshape(-1, 1)

        # Solve the linear system x = lhs \ rhs
        solution = np.linalg.solve(lhs, rhs)

        #
        self.coefficients = solution.reshape((numberOfContrasts, numberOf3DBasisFunctions)).T

    def setBiasFieldCoefficients(self, coefficients):
        if coefficients is not None:
            self.coefficients = coefficients.cuda()
        else:
            self.coefficients = coefficients

    def downSampleBasisFunctions(self, downSamplingFactors):
        self.basisFunctions = [np.tensor(biasFieldBasisFunction[::downSamplingFactor])
                                        for biasFieldBasisFunction, downSamplingFactor in
                                        zip(self.fullBasisFunctions, downSamplingFactors)]
