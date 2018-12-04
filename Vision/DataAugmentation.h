#pragma once

#include "AllIncludes.h"

/*	
	Documentation
		- https://github.com/karanshgl/image-operations/blob/master/main.cpp#L264
*/

enum AugmentationType {
	DISTORTION,
	TRANSLATION,
	ROTATION,
	FLIPPING,
	NOISE,
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
	static cv::Mat GetAugmentedFrame(cv::Mat &input);
	~DataAugmentation();
private:
	// Types of transformations
	static void Distortion(cv::Mat &input);
	static void Translate(cv::Mat &input);
	static void Rotate(cv::Mat &input);
	static void Flip(cv::Mat &input);
	static void SaltNPepper(cv::Mat &input);
	static void Noise(cv::Mat &input);
	static void Lightning(cv::Mat &input);
	static void Perspective(cv::Mat &input);

	// General variables
	cv::Mat _frame, _reshapedFrame;
	AugmentationType _transformation;
};

