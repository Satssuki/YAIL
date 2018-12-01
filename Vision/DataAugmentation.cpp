#include "DataAugmentation.h"

DataAugmentation::DataAugmentation(cv::Mat input) {
	_frame = input;
	_transformation = (AugmentationType)(rand() % PERSPECTIVE);
}

cv::Mat DataAugmentation::GetAugmentedFrame()
{
	switch (_transformation) {
	/*case (NONE):
		return _frame;
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
		return this->Perspective();*/
	default:
		return this->Scale();
	}
}

DataAugmentation::~DataAugmentation() {}

#pragma region Scaling transformations
bool DataAugmentation::inRange(int x, int be, int en)
{
	return (be <= x && x < en);
}

cv::Mat DataAugmentation::Scale()
{
	int scale = rand() % 3;

	int nRows = _frame.rows;
	int nCols = _frame.cols;
	int channels = _frame.channels();

	int newRow = floor(1 * nRows);
	int newCol = floor(1 * nCols);

	_reshapedFrame = cv::Mat(newRow, newCol, CV_8UC3);

	double sr = (nRows - 1)*1.0 / (newRow - 1);
	double sc = (nCols - 1)*1.0 / (newCol - 1);

	cv::Vec3b *p, *q;

	for (int i = 0; i < newRow; i++) {
		p = _frame.ptr<cv::Vec3b>(i);
		q = _reshapedFrame.ptr<cv::Vec3b>(i);
		for (int j = 0; j < newCol; j++) {
			for (int k = 0; k < channels; k++) {
				int x = round(i / scale);
				int y = round(i / scale);
				
				if (inRange(x, 0, nRows) && inRange(y, 0, nCols)) {
					q[j][k] = _frame.at<cv::Vec3b>(x, y).val[k];
				}
			}
		}
	}
	return _reshapedFrame;
}
#pragma endregion

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
