#include "Function.h"


Eigen::VectorXf Function::ActivationFunction(Activation activation, Eigen::VectorXf  src)
{
	switch (activation)
	{
	case sigmoid:
		src = Sigmoid(src);
		break;

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



