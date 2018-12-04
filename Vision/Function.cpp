#include "Function.h"


Eigen::VectorXf Function::ActivationFunc(Activation activationFunc, Eigen::VectorXf  src)
{
	switch (activationFunc)
	{
	case sigmoid:
		src = Sigmoid(src);
		break;

	default:
			break;
	}
	return src;
}

Eigen::VectorXf Function::Sigmoid(Eigen::VectorXf  src)
{
	return 1.0 / (1.0 + src.exp() * -1);
}



