import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

images = np.loadtxt("faces.dat")  # raw 400 * 4096 pixel data? Need to reshape
imagesParsed = np.reshape(images, (400, 64, 64))

# Question 3a
image100 = imagesParsed[99]
plt.imshow(image100)    # plotting the 100th image
plt.show()


# Question 3b
# imagesMean = np.mean(images, axis=0)
# imagesMean = np.reshape(imagesMean, (1, 4096))
# removing the mean of each column from the feature set
# imagesMinusMean = images - imagesMean
# imagesMinusMeanParsed = np.reshape(imagesMinusMean, (400, 64, 64))
#
# plt.imshow(imagesMinusMeanParsed[99])
# plt.show()


# Question 3c
# imagesMean = np.mean(images, axis=0)
# imagesMean = np.reshape(imagesMean, (1, 4096))
# imagesMinusMean = images - imagesMean
#
# myPCA = PCA()
# myPCA.fit(imagesMinusMean)
# myEigenvalues = np.sort(myPCA.explained_variance_)   # in ascending! Must reverse the order now
# myDescendingEigenvalues = myEigenvalues[::-1]
#
# plt.figure(figsize=(10,7))
# plt.scatter(np.linspace(0, 400, 400), myDescendingEigenvalues)
# plt.xlabel("Components")
# plt.ylabel("Eigenvalues")
#
# plt.tight_layout()
# plt.show()


# Question 3e
# imagesMean = np.mean(images, axis=0)
# imagesMean = np.reshape(imagesMean, (1, 4096))
# imagesMinusMean = images - imagesMean
#
# myPCA = PCA()
# myPCA.fit(imagesMinusMean)
# myEigenvalues = np.sort(myPCA.explained_variance_)
# myDescendingEigenvalues = myEigenvalues[::-1]
# myVariances = np.sort(myPCA.explained_variance_ratio_)
# myDescendingVariances = myVariances[::-1]
#
# totalVariance = 0  # aiming to explain 95% of variance
# numComponents = 0
# while totalVariance < 0.95:
#     totalVariance += myDescendingVariances[numComponents]
#     numComponents += 1
#
# print(numComponents, "components needed to explain 95% of variance")
#
# componentList = []
# componentList.extend(range(1,101))
# variancesList = []
# for i in range(0, 100):
#     variancesList.append(myDescendingVariances[i])
#
# print(len(componentList), len(variancesList))
# plt.figure(figsize=(10, 7))
# plt.bar(componentList, variancesList)
# plt.xlabel("Components")
# plt.ylabel("Variance (%)")
#
# plt.tight_layout()
# plt.show()


# Question 3f
# imagesMean = np.mean(images, axis=0)
# imagesMean = np.reshape(imagesMean, (1, 4096))
# imagesMinusMean = images - imagesMean
#
# myPCA = PCA()
# myPCA.fit(imagesMinusMean)  # perform PCA process on the image data
# myEigenvectors = np.sort(myPCA.components_)
# myDescendingEigenvectors = myEigenvectors[::-1] # get top five eigenvectors from here
# topFiveEigenvectors = myPCA.components_[0:5]
# print(topFiveEigenvectors)
#
# transformedImages = imagesMinusMean @ topFiveEigenvectors.T
# newImages = transformedImages @ topFiveEigenvectors # transform back to original space
# newImagesParsed = np.reshape(newImages, (400, 64, 64))
#
# rows = 1;
# columns = 5;
# figure = plt.figure(figsize=(8,8))
#
# for i in range(1, columns*rows + 1):
#     image = newImagesParsed[i-1]
#     figure.add_subplot(rows, columns, i)
#     plt.imshow(image)
#
# plt.tight_layout()
# plt.show()

# Question 3g. I just changed the n_components parameter each time...
# imagesMean = np.mean(images, axis=0)
# print(imagesMean.shape)
# imagesMean = np.reshape(imagesMean, (1, 4096))
# imagesMinusMean = images - imagesMean
#
# myPCA = PCA(n_components=399)
# myPCA.fit(imagesMinusMean)
# imagesPCA = myPCA.transform(imagesMinusMean)
# newImages = myPCA.inverse_transform(imagesPCA)
# print("New images shape: ", newImages.shape)
# newImagesParsed = np.reshape(newImages, (400, 64, 64))
#
# plt.imshow(newImagesParsed[99])
#
# plt.tight_layout()
# plt.show()