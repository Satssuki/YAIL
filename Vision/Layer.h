#pragma once

#ifndef LAYER_H
#define LAYER_H

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <string>

#include "Activation.h"

class Layer
{
public:
	Layer();
	// dense
	Layer(int neurons, Activation activation);
	
	// input 
	Layer(int neurons);

	// conv2d
	Layer(int countFilters, cv::Size filterSize, cv::Size imageSize, Activation activation);

	// maxpool2d
	Layer(cv::Size poolSize);
	
	~Layer();

	int Size();
	virtual std::string ToString();
	Activation _Activation;
	
	// conv2d
	int CountFilters;
	cv::Size FilterSize;
	cv::Size ImageSize;

	// maxpool2d
	cv::Size PoolSize;

private:
	// dense
	int Neurons;
	
};

#endif 