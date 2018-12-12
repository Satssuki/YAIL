#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

#include <iostream>
#include <string>

#include "Activation.h"

class Layer
{
public:
	Layer();
	Layer(int neurons, Activation activation);
	Layer(int neurons);
	~Layer();

	int Size();
	virtual std::string ToString();
	Activation _Activation;
private:
	int Neurons;
};

#endif 