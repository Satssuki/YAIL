#pragma once
#include <Eigen/Dense>
#include "Activation.h"

class Function
{
public:

	static Eigen::VectorXf ActivationFunc(Activation activationFunc, Eigen::VectorXf  src);

private:
	static Eigen::VectorXf Sigmoid(Eigen::VectorXf  src);
};

