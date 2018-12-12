#pragma once

#ifndef CONV_H
#define CONV_H

#include "Layer.h"

namespace Layers {
	class Conv2D : public Layer
	{
	public:
		Conv2D();
		Conv2D(int countFilters, cv::Size filterSize, cv::Size imageSize, Activation activation) :Layer(countFilters, filterSize, imageSize, activation) {};
		~Conv2D();
	};
}
#endif

