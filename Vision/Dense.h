#pragma once
#include "AllIncludes.h"

namespace Layers {
	class Dense : public Layer
	{
	public:
		Dense();
		Dense(int neurons, std::string activation) :Layer(neurons, activation) {};
		Dense(int neurons, Activation activation) :Layer(neurons, activation) {};
		~Dense();
	};
}


