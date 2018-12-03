#pragma once
#include "AllIncludes.h"

class Layer
{
public:
	Layer();
	Layer(int totalNeurons, std::string activationFunc);
	~Layer();
};

