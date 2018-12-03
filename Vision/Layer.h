#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <string>

#include "Activation.h"

class Layer
{
public:
	Layer();
	Layer(int neurons, std::string activation);
	Layer(int neurons, Activation activation);
	~Layer();
};

#endif 