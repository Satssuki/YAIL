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

Eigen::VectorXf Function::ErrorFunction(Loss loss, Eigen::VectorXf src)
{
	switch (loss)
	{
	case mean_squared_error:
		src = MeanSquaredError(src);
		break;

	default:
		break;
	}
	return src;
}


Eigen::VectorXf Function::Sigmoid(Eigen::VectorXf  src)
{
	return Eigen::VectorXf();
	//return 1.0 / (1.0 + src.exp() * -1);
}

Eigen::VectorXf Function::MeanSquaredError(Eigen::VectorXf src)
{
	return Eigen::VectorXf();
}



