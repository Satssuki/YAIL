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
	case cross_entropy:
		output = CrossEntropyError(y, output);
		break;
	default:
		break;
	}
	return output;
}

Eigen::VectorXf Function::DeltaLastLayer(Loss loss, Eigen::VectorXf y, Eigen::VectorXf output, Eigen::VectorXf z)
{
	switch (loss)
	{
	case mean_squared_error:
		output = MeanSquaredErrorPrime(y, output).array() * SigmoidPrime(z).array();
		break;
	case cross_entropy:
		output = CrossEntropyErrorPrime(y, output);
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

Eigen::VectorXf Function::CrossEntropyError(Eigen::VectorXf y, Eigen::VectorXf output)
{
	//float verif =
	Eigen::VectorXf crossEntropy = (-1 * y.array()) * log(output.array()) - (1 - y.array()) * log(1 - output.array());
	
	if (crossEntropy.hasNaN())
	{
		for (int i = 0; i < crossEntropy.rows(); i++)
		{
			if (isnan(crossEntropy(i)))
				crossEntropy(i) = 0;
		}
	}

	return crossEntropy;
}

Eigen::VectorXf Function::SigmoidPrime(Eigen::VectorXf src)
{
	return Sigmoid(src).array() * (1 - Sigmoid(src).array());
}

// add the rest of the code lol, en ce moment il est pas placé a la bonne place
Eigen::VectorXf Function::MeanSquaredErrorPrime(Eigen::VectorXf y, Eigen::VectorXf output)
{
	return output.array() - y.array();
}

Eigen::VectorXf Function::CrossEntropyErrorPrime(Eigen::VectorXf y, Eigen::VectorXf output)
{
	return output.array() - y.array();
}



