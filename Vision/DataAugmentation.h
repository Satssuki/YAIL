#pragma once

#include "AllIncludes.h"

/*	
	Documentation
		- https://github.com/karanshgl/image-operations/blob/master/main.cpp#L264
*/

enum AugmentationType {
	NONE,
	DISTORTION,
	TRANSLATION,
	ROTATION,
	FINER_ROTATION,
	FLIPPING,
	PEPPER_AND_SALT,
	LIGHTNING,
	PERSPECTIVE,
};

enum Interpolation {
	nearest,
	bilinear,
};

class DataAugmentation
{

public:
	DataAugmentation(cv::Mat input);
	cv::Mat GetAugmentedFrame();
	~DataAugmentation();
private:
	// Useful functions
	bool inRange(int x, int be, int en);
	// Types of transformations
	cv::Mat Distortion();
	cv::Mat Translate();
	cv::Mat Rotate();
	cv::Mat Fine_Rotate();
	cv::Mat Flip();
	cv::Mat Noise();
	cv::Mat Lightning();
	cv::Mat Perspective();

	// General variables
	cv::Mat _frame, _reshapedFrame;
	AugmentationType _transformation;
};

