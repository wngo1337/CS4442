import numpy as np
import matplotlib.pyplot as plt

# CS4442 Assignment 1
# Code written by William Ngo

# A lot of the code is duplicated because I wanted to test each question separately...
# Probably should have turned a lot of this into functions, but I ran out of time

xTrainingData = np.loadtxt("./hw1xtr.dat.txt") # load training data
yTrainingData = np.loadtxt("./hw1ytr.dat.txt")

xTestData = np.loadtxt("./hw1xte.dat.txt")  # load test data
yTestData = np.loadtxt("./hw1yte.dat.txt")

# fig, (ax1, ax2), = plt.subplots(nrows=2,ncols=1)  # setting up figure for graphs
#
# ax1.scatter(xTrainingData, yTrainingData)
# ax1.set_title("Question 2 Training Set")
# ax1.set_xlabel("x-axis")
# ax1.set_ylabel("y-axis")
#
#
# ax2.scatter(xTestData, yTestData)
# ax2.set_title("Question 2 Test Set")
# ax2.set_xlabel("x-axis")
# ax2.set_ylabel("y-axis")
#
# plt.tight_layout()
# plt.show()  # end Question 2, part a



# Start question 2 part b
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
# yOnes = np.ones((len(modifiedYTraining), 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# intercept, slope = weight[0], weight[1]
#
#
# plt.scatter(xTrainingData, yTrainingData)
# plt.title("Training Data Regression Line")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# xInterval = np.linspace(0, max(xTrainingData), xTrainingData.size)
#
# yValue = slope * xInterval + intercept
# plt.plot(xInterval, yValue)
#
# error = 0;
# for i in range(xTrainingData.size): # calculating the error
#     error += np.square((slope * xTrainingData[i] + intercept) - yTrainingData[i])
#
# error = (1/xTrainingData.size) * error
# print("the error is ", error)
# plt.tight_layout
# plt.show()  # end question 2b



# # start question 2c
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
# yOnes = np.ones((len(modifiedYTraining), 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# intercept, slope = weight[0], weight[1]
#
# plt.scatter(xTestData, yTestData)
# plt.title("Test Data")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# xInterval = np.linspace(0, max(xTestData), xTestData.size)
#
# yValue = slope * xInterval + intercept
# plt.plot(xInterval, yValue)
#
# error = 0;
# for i in range(xTestData.size): # calculating the error
#     error += np.square((slope * xTestData[i] + intercept) - yTestData[i])
#
# error = (1/xTestData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()  # end question 2c



# # start question 2d
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData) # now have our x^2 values to augment
# xSquared = np.reshape(xSquared, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# # this time, coefficients are given in terms of the intercept, the linear, and then the quadratic
# constantValue, linearValue, quadraticValue = weight[0], weight[1], weight[2]
#
# plt.scatter(xTrainingData, yTrainingData)
# xInterval = np.linspace(0, max(xTrainingData), xTrainingData.size)
#
# yValue = quadraticValue * np.square(xInterval) + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Training Data (Quadratic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTrainingData.size): # calculating the error
#     xVal = xTrainingData[i]
#     error += np.square((quadraticValue * np.square(xVal) + linearValue * xVal + constantValue)
#                        - yTrainingData[i])
#
# error = (1/xTrainingData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # start question 2d with test data
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData) # now have our x^2 values to augment
# xSquared = np.reshape(xSquared, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# # this time, coefficients are given in terms of the intercept, the linear, and then the quadratic
# constantValue, linearValue, quadraticValue = weight[0], weight[1], weight[2]
#
# plt.scatter(xTestData, yTestData)
# xInterval = np.linspace(0, max(xTestData), xTestData.size)
#
# yValue = quadraticValue * np.square(xInterval) + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Test Data (Quadratic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTestData.size): # calculating the error
#     xVal = xTestData[i]
#     error += np.square((quadraticValue * np.square(xVal) + linearValue * xVal + constantValue)
#                        - yTestData[i])
#
# error = (1/xTestData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # start question 2e
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData)
# xSquared = np.reshape(xSquared, (-1, 1))
#
# xCubed = np.power(xTrainingData, 3)   # need cubed values for cubic regression
# xCubed = np.reshape(xCubed, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# constantValue, linearValue, quadraticValue , cubicValue = weight[0], weight[1], weight[2], weight[3]
#
# plt.scatter(xTrainingData, yTrainingData)
# xInterval = np.linspace(0, max(xTrainingData), xTrainingData.size)
#
# yValue = cubicValue * np.power(xInterval, 3) + quadraticValue * np.square(xInterval) \
#          + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Training Data (Cubic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTrainingData.size): # calculating the error
#     xVal = xTrainingData[i]
#     xApprox = cubicValue * np.power(xVal, 3) + quadraticValue * np.square(xVal) \
#               + linearValue * xVal + constantValue
#     error += np.square(xApprox - yTrainingData[i])
#
# error = (1/xTrainingData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # start question 2e test
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData)
# xSquared = np.reshape(xSquared, (-1, 1))
#
# xCubed = np.power(xTrainingData, 3)
# xCubed = np.reshape(xCubed, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed), axis=1)
#
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# # this time, coefficients are given in terms of the intercept, the linear, and then the quadratic
# constantValue, linearValue, quadraticValue , cubicValue = weight[0], weight[1], weight[2], weight[3]
#
# plt.scatter(xTestData, yTestData)
# xInterval = np.linspace(0, max(xTestData), xTestData.size)
#
# yValue = cubicValue * np.power(xInterval, 3) + quadraticValue * np.square(xInterval) \
#          + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Test Data (Cubic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTestData.size): # calculating the error
#     xVal = xTestData[i]
#     xApprox = cubicValue * np.power(xVal, 3) + quadraticValue * np.square(xVal) \
#               + linearValue * xVal + constantValue
#     error += np.square(xApprox - yTestData[i])
#
# error = (1/xTestData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # Start question 2f
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData)
# xSquared = np.reshape(xSquared, (-1, 1))
#
# xCubed = np.power(xTrainingData, 3)
# xCubed = np.reshape(xCubed, (-1, 1))
#
# xQuad = np.power(xTrainingData, 4)
# xQuad = np.reshape(xQuad, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed, xQuad), axis=1)
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# constantValue, linearValue, quadraticValue, cubicValue, quarticValue = \
#     weight[0], weight[1], weight[2], weight[3], weight[4]
#
# plt.scatter(xTrainingData, yTrainingData)
# xInterval = np.linspace(0, max(xTrainingData), xTrainingData.size)
#
# yValue = quarticValue * np.power(xInterval, 4) + \
#          cubicValue * np.power(xInterval, 3) + quadraticValue * np.square(xInterval) \
#          + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Training Data (Quartic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTrainingData.size): # calculating the error
#     xVal = xTrainingData[i]
#     xApprox = quarticValue * np.power(xVal, 4) + \
#               cubicValue * np.power(xVal, 3) + quadraticValue * np.square(xVal) \
#               + linearValue * xVal + constantValue
#     error += np.square(xApprox - yTrainingData[i])
#
# error = (1/xTrainingData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # start question 2f test data
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData)
# xSquared = np.reshape(xSquared, (-1, 1))
#
# xCubed = np.power(xTrainingData, 3)
# xCubed = np.reshape(xCubed, (-1, 1))
#
# xQuad = np.power(xTrainingData, 4)
# xQuad = np.reshape(xQuad, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed, xQuad), axis=1)
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
#
# xTranspose = modifiedXTraining.T
#
# weight = np.dot(np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining)), xTranspose),
#                  modifiedYTraining) # calculating the values of the regression line
# constantValue, linearValue, quadraticValue, cubicValue, quarticValue = \
#     weight[0], weight[1], weight[2], weight[3], weight[4]
#
# plt.scatter(xTestData, yTestData)
# xInterval = np.linspace(0, max(xTestData), xTestData.size)
#
# yValue = quarticValue * np.power(xInterval, 4) + \
#          cubicValue * np.power(xInterval, 3) + quadraticValue * np.square(xInterval) \
#          + linearValue * xInterval + constantValue
# plt.plot(xInterval, yValue)
# plt.title("Test Data (Quartic Regression)")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
#
# error = 0;
# for i in range(xTestData.size): # calculating the error
#     xVal = xTestData[i]
#     xApprox = quarticValue * np.power(xVal, 4) + \
#               cubicValue * np.power(xVal, 3) + quadraticValue * np.square(xVal) \
#               + linearValue * xVal + constantValue
#     error += np.square(xApprox - yTestData[i])
#
# error = (1/xTestData.size) * error
# print("the error is ", error)
#
# plt.tight_layout
# plt.show()



# # start 3a
# modifiedXTraining = xTrainingData.copy()
# modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
# xOnes = np.ones((len(modifiedXTraining), 1))
# modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
# xSquared = np.square(xTrainingData)
# xSquared = np.reshape(xSquared, (-1, 1))
#
# xCubed = np.power(xTrainingData, 3)
# xCubed = np.reshape(xCubed, (-1, 1))
#
# xQuad = np.power(xTrainingData, 4)
# xQuad = np.reshape(xQuad, (-1, 1))
#
# modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed, xQuad), axis=1)
# #   Have now added column vector of 1s to the x data)
# modifiedYTraining = yTrainingData.copy()    # converted data into column vector
# modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
# xTranspose = modifiedXTraining.T
#
#
# # Now just need to create an identity matrix to multiply with regularization parameter
# identityMatrix = np.identity(5)
# identityMatrix[0][0] = 0    # Don't regularize W0?
# LAMBDA_LIST = [0.01, 0.1, 1, 10, 100, 1000, 10000]
#
# for param in LAMBDA_LIST:   # compute error for each lambda
#     weight = np.dot(
#         np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining) + param * identityMatrix), xTranspose),
#         modifiedYTraining)
#     constantValue, linearValue, quadraticValue, cubicValue, quarticValue = \
#         weight[0], weight[1], weight[2], weight[3], weight[4]
#
#     error = 0
#     for i in range(xTrainingData.size):
#         xTrain = xTrainingData[i]
#         yTrain = yTrainingData[i]
#         xTrainApprox = quarticValue * np.power(xTrain, 4) + \
#                   cubicValue * np.power(xTrain, 3) + quadraticValue * np.square(xTrain) \
#                   + linearValue * xTrain + constantValue
#         error += np.square(xTrainApprox - yTrain)
#
#     for j in range(xTestData.size):
#         xTest = xTestData[j]
#         yTest = yTestData[j]
#         xTestApprox = quarticValue * np.power(xTest, 4) + \
#                   cubicValue * np.power(xTest, 3) + quadraticValue * np.square(xTest) \
#                   + linearValue * xTest + constantValue
#         error += np.square(xTestApprox - yTest)
#     error = 1/(xTrainingData.size + xTestData.size) * error
#
#     plt.scatter(param, error)
#
# plt.title("Error For Given Lambda Values")
# plt.xlabel("Lambda values")
# plt.ylabel("Error")
# plt.xscale("log", base=10)
# plt.tight_layout()
# plt.show()



# start 3b
modifiedXTraining = xTrainingData.copy()
modifiedXTraining = np.reshape(modifiedXTraining, (-1, 1))
xOnes = np.ones((len(modifiedXTraining), 1))
modifiedXTraining = np.concatenate((xOnes, modifiedXTraining), axis=1)
xSquared = np.square(xTrainingData)
xSquared = np.reshape(xSquared, (-1, 1))

xCubed = np.power(xTrainingData, 3)
xCubed = np.reshape(xCubed, (-1, 1))

xQuad = np.power(xTrainingData, 4)
xQuad = np.reshape(xQuad, (-1, 1))

modifiedXTraining = np.concatenate((modifiedXTraining, xSquared, xCubed, xQuad), axis=1)
#   Have now added column vector of 1s to the x data)
modifiedYTraining = yTrainingData.copy()    # converted data into column vector
modifiedYTraining = np.reshape(modifiedYTraining, (-1, 1))
xTranspose = modifiedXTraining.T

# Now just need to create an identity matrix to multiply with regularization parameter
identityMatrix = np.identity(5)
identityMatrix[0][0] = 0    # Don't regularize W0?
LAMBDA_LIST = [0.01, 0.1, 1, 10, 100, 1000, 10000]
constantWeights = []
linearWeights = []
quadraticWeights = []
cubicWeights = []
quarticWeights = []

for param in LAMBDA_LIST:   # compute error for each lambda
    weightCounter = 0
    weight = np.dot(
        np.dot(np.linalg.inv(np.dot(xTranspose, modifiedXTraining) + param * identityMatrix), xTranspose),
        modifiedYTraining)
    constantValue, linearValue, quadraticValue, cubicValue, quarticValue = \
        weight[0], weight[1], weight[2], weight[3], weight[4]

    weightCounter = 0;
    for i in range(weight.size):
        weightValue = weight[i]
        plt.scatter(param, weight[i])
        if weightCounter == 0:  # this is horrible, but I ran out of time
            constantWeights.append(weightValue)
        elif weightCounter == 1:
            linearWeights.append(weightValue)
        elif weightCounter == 2:
            quadraticWeights.append(weightValue)
        elif weightCounter == 3:
            cubicWeights.append(weightValue)
        elif weightCounter == 4:
            quarticWeights.append(weightValue)
        weightCounter += 1

print(constantWeights)
plt.plot(LAMBDA_LIST, constantWeights)
plt.plot(LAMBDA_LIST, linearWeights)
plt.plot(LAMBDA_LIST, quadraticWeights)
plt.plot(LAMBDA_LIST, cubicWeights)
plt.plot(LAMBDA_LIST, quarticWeights)

plt.title("Weights For Given Lambda Values")
plt.xlabel("Lambda values")
plt.ylabel("Weight Parameters")
plt.xscale("log", base=10)
plt.show()