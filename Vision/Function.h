#pragma once
#include <Eigen/Dense>
#include "Activation.h"
#include "Loss.h"
#include <iostream>

class Function
{
public:
	static Eigen::VectorXf ActivationFunction(Activation activation, Eigen::VectorXf src);
	static Eigen::VectorXf ErrorFunction(Loss loss, Eigen::VectorXf y, Eigen::VectorXf output);
	
	static Eigen::VectorXf ActivationFunctionPrime(Activation activation, Eigen::VectorXf src);

	static Eigen::VectorXf DeltaLastLayer(Loss loss, Eigen::VectorXf y, Eigen::VectorXf output, Eigen::VectorXf z);
private:
	// activation
	static Eigen::VectorXf Sigmoid(Eigen::VectorXf src);

	// loss
	static Eigen::VectorXf MeanSquaredError(Eigen::VectorXf y, Eigen::VectorXf output);
	static Eigen::VectorXf CrossEntropyError(Eigen::VectorXf y, Eigen::VectorXf output);

	// activation prime
	static Eigen::VectorXf SigmoidPrime(Eigen::VectorXf src);

	// loss prime
	static Eigen::VectorXf MeanSquaredErrorPrime(Eigen::VectorXf y, Eigen::VectorXf output);

	//Relu
	static Eigen::VectorXf LeakyRelu(Eigen::VectorXf src);

	//Relu prime
	static Eigen::VectorXf LeakyReluPrime(Eigen::VectorXf src);


	//SoftMax
	static Eigen::VectorXf SoftMax(Eigen::VectorXf src);

	static Eigen::VectorXf SoftMaxPrime(Eigen::VectorXf src);

	static Eigen::VectorXf CrossEntropyErrorPrime(Eigen::VectorXf y, Eigen::VectorXf output);

};

