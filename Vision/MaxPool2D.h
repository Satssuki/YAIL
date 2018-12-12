#pragma once

#ifndef MAXPOOL_H
#define MAXPOOL_H

#include "Layer.h"

namespace Layers {
	class MaxPool2D : public Layer
	{
	public:
		MaxPool2D();
		MaxPool2D(cv::Size poolSize) :Layer(poolSize) {};
		~MaxPool2D();
	};
}
#endif

