#pragma once

#ifndef DENSE_H
#define DENSE_H

#include "Layer.h"

namespace Layers {
	class Dense : public Layer
	{
	public:
		Dense();
		Dense(int neurons, Activation activation) :Layer(neurons, activation) {};
		Dense(int neurons, std::string activation) :Layer(neurons, activation) {};
		~Dense();

		std::string ToString();
	};
}
#endif

