#include "DataAugmentation.h"

DataAugmentation::DataAugmentation(cv::Mat input) {
	_frame = input;
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
