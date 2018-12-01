#include "DataAugmentation.h"

DataAugmentation::DataAugmentation(cv::Mat input) {
	_frame = input;
	_transformation = (AugmentationType)(rand() % PERSPECTIVE);
	int x;
}

cv::Mat DataAugmentation::GetAugmentedFrame()
{
	switch (_transformation) {
	case (NONE):
		return _reshapedFrame;
	case (TRANSLATION):
		return this->Translate();
	case (ROTATION):
		return this->Rotate();
	case (FINER_ROTATION):
		return this->Fine_Rotate();
	case (FLIPPING):
		return this->Flip();
	case (PEPPER_AND_SALT):
		return this->Noise();
	case (LIGHTNING):
		return this->Lightning();
	case (PERSPECTIVE):
		return this->Perspective();
	default:
		return _reshapedFrame;
	}
}

DataAugmentation::~DataAugmentation() {}

cv::Mat DataAugmentation::Scale()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Translate()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Rotate()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Fine_Rotate()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Flip()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Noise()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Lightning()
{
	return cv::Mat();
}

cv::Mat DataAugmentation::Perspective()
{
	return cv::Mat();
}
