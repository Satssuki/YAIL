#include "Function.h"


Eigen::VectorXf Function::ActivationFunction(Activation activation, Eigen::VectorXf  src)
{
	switch (activation)
	{
	case sigmoid:
		src = Sigmoid(src);
		break;
	case leakyRelu:
		src = LeakyRelu(src);
	default:
			break;
	}
	return src;
}

Eigen::VectorXf Function::ErrorFunction(Loss loss, Eigen::VectorXf y, Eigen::VectorXf output)
{
	switch (loss)
	{
	case mean_squared_error:
		output = MeanSquaredError(y, output);
		break;

	default:
		break;
	}
	return output;
}

Eigen::VectorXf Function::ErrorFunctionPrime(Loss loss, Eigen::VectorXf y, Eigen::VectorXf output)
{
	switch (loss)
	{
	case mean_squared_error:
		output = MeanSquaredErrorPrime(y, output);
		break;

	default:
		break;
	}
	return output;
}

Eigen::VectorXf Function::ActivationFunctionPrime(Activation activation, Eigen::VectorXf src)
{
	switch (activation)
	{
	case sigmoid:
		src = SigmoidPrime(src);
		break;
	case leakyRelu : 
		src = LeakyReluPrime(src);
	default:
		break;
	}
	return src;
}


Eigen::VectorXf Function::Sigmoid(Eigen::VectorXf  src)
{
	return 1.0 / (1 + exp(-1 * src.array()));
}

Eigen::VectorXf Function::MeanSquaredError(Eigen::VectorXf y, Eigen::VectorXf output)
{
	return (y.array() - output.array()).pow(2) / 2.0;
}

Eigen::VectorXf Function::SigmoidPrime(Eigen::VectorXf src)
{
	return Sigmoid(src).array() * (1 - Sigmoid(src).array());
}

Eigen::VectorXf Function::MeanSquaredErrorPrime(Eigen::VectorXf y, Eigen::VectorXf output)
{
	return output.array() - y.array();
}

Eigen::VectorXf Function::LeakyRelu(Eigen::VectorXf v)
{

	Eigen::VectorXf activation(v.rows(), v.cols());
	//If the vector is 2d
	for (int i = 0; i < v.rows(); i++)
	{
		for (int j = 0; j < v.cols(); j++) 
		{
			if (v(i, j) <= 0.0f) 
			{
				activation(i, j) = v(i,j) * 0.05;
			}
			else {
				activation(i, j) = v(i,j);
			}
		}
	}
	return activation;
}

Eigen::VectorXf Function::LeakyReluPrime(Eigen::VectorXf v)
{

	Eigen::VectorXf activationPrime(v.rows(), v.cols());
	//If the vector is 2d
	for (int i = 0; i < v.rows(); i++)
	{
		for (int j = 0; j < v.cols(); j++)
		{
			if (v(i, j) <= 0.0f)
			{
				activationPrime(i, j) = (0.05);
			}
			else {
				activationPrime(i, j) = (1.0);
			}
		}
	}

	return activationPrime;
}





