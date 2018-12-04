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
	case (DISTORTION):
		return this->Distortion();
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
		return this->Translate();
	}
}

DataAugmentation::~DataAugmentation() {}

cv::Mat DataAugmentation::Distortion()
{
	_reshapedFrame = _frame.clone();
	double sigma = 4.0;
	double alpha = 100;
	bool bNorm = false; _reshapedFrame.clone();


	cv::Mat dx(_frame.size(), CV_64FC1);
	cv::Mat dy(_frame.size(), CV_64FC1);

	double low = -1.0;
	double high = 1.0;

	//The image deformations were created by first generating
	//random displacement fields, that's dx(x,y) = rand(-1, +1) and dy(x,y) = rand(-1, +1)
	cv::randu(dx, cv::Scalar(low), cv::Scalar(high));
	cv::randu(dy, cv::Scalar(low), cv::Scalar(high));

	//The fields dx and dy are then convolved with a Gaussian of standard deviation sigma(in pixels)
	cv::Size kernel_size(sigma * 6 + 1, sigma * 6 + 1);
	cv::GaussianBlur(dx, dx, kernel_size, sigma);
	cv::GaussianBlur(dy, dy, kernel_size, sigma);

	//If we normalize the displacement field (to a norm of 1,
	//the field is then close to constant, with a random direction
	if (bNorm)
	{
		dx /= cv::norm(dx, cv::NORM_L1);
		dy /= cv::norm(dy, cv::NORM_L1);
	}

	//The displacement fields are then multiplied by a scaling factor alpha
	//that controls the intensity of the deformation.
	dx *= alpha;
	dy *= alpha;

	//Inverse(or Backward) Mapping to avoid gaps and overlaps.
	cv::Rect checkError(0, 0, _frame.cols, _frame.rows);
	int nCh = _frame.channels();

	for (int displaced_y = 0; displaced_y < _frame.rows; displaced_y++)
		for (int displaced_x = 0; displaced_x < _frame.cols; displaced_x++)
		{
			int org_x = displaced_x - dx.at<double>(displaced_y, displaced_x);
			int org_y = displaced_y - dy.at<double>(displaced_y, displaced_x);

			if (checkError.contains(cv::Point(org_x, org_y)))
			{
				for (int ch = 0; ch < nCh; ch++)
				{
					_reshapedFrame.data[(displaced_y * _frame.cols + displaced_x) * nCh + ch] = _frame.data[(org_y * _frame.cols + org_x) * nCh + ch];
				}
			}
		}
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Translate()
{
	float offsetX = rand() % 250;
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, 120, 0, 1, -100);
	cv::warpAffine(_frame, _reshapedFrame, trans_mat, _frame.size());
	return _reshapedFrame;
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
