#pragma once

#include "AllIncludes.h"

enum AugmentationType {
	NONE,
	SCALING,
	TRANSLATION,
	ROTATION,
	FINER_ROTATION,
	FLIPPING,
	PEPPER_AND_SALT,
	LIGHTNING,
	PERSPECTIVE,
};
class DataAugmentation
{

public:
	DataAugmentation(cv::Mat input);
	~DataAugmentation();
private:
	cv::Mat Scale();
	cv::Mat Translate();
	cv::Mat Rotate();
	cv::Mat Fine_Rotate();
	cv::Mat Flip();
	cv::Mat Noise();
	cv::Mat Lightning();
	cv::Mat Perspective();
	cv::Mat _frame;
	AugmentationType _transformation;
};

