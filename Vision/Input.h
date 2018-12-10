#pragma once

#ifndef INPUT_H
#define INPUT_H

#include "Layer.h"

namespace Layers {
	class Input : public Layer
	{
	public:
		Input();
		Input(int neurons) :Layer(neurons) {};
		~Input();

		std::string ToString();
	};
}
#endif