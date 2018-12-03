#pragma once
#include "AllIncludes.h"

class Layer
{
public:
	Layer();
	Layer(int neurons, std::string activation);
	Layer(int neurons, Activation activation);
	~Layer();
};

