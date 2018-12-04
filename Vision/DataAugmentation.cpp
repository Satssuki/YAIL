#include "DataAugmentation.h"

DataAugmentation::DataAugmentation(cv::Mat input) {
	_frame = input;
	_transformation = (AugmentationType)(rand() % PERSPECTIVE);
}

cv::Mat DataAugmentation::GetAugmentedFrame()
{
	switch (_transformation) {
	case (NONE):
		return _frame;
	case (DISTORTION):
		return this->Distortion();
	case (TRANSLATION):
		return this->Translate();
	case (ROTATION):
		return this->Rotate();
	case (FLIPPING):
		return this->Flip();
	case (PEPPER_AND_SALT):
		return this->Noise();
	case (LIGHTNING):
		return this->Lightning();
	case (PERSPECTIVE):
		return this->Perspective();
	default:
		return _frame;
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
	float offsetY = rand() % 100;
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetX, 0, 1, offsetY);
	cv::warpAffine(_frame, _reshapedFrame, trans_mat, _frame.size());
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Rotate()
{
	cv:: Point2f src_center(_frame.cols / 2.0F, _frame.rows / 2.0F);
	cv::Mat rot_mat = getRotationMatrix2D(src_center, rand() % 180, 1.0);
	cv::warpAffine(_frame, _reshapedFrame, rot_mat, _frame.size());
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Flip()
{
	cv::flip(_frame, _reshapedFrame, 1);
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Noise()
{
	_reshapedFrame = _frame.clone();
	cv::blur(_frame, _reshapedFrame, cv::Size(20, 20));
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Lightning()
{
	_reshapedFrame = _frame.clone();
	for (int rows = 0; rows < _frame.rows; rows++) {
		for (int cols = 0; cols < _frame.cols; cols++) {
			_reshapedFrame.at<cv::Vec3b>(rows, cols)[0] = _frame.at<cv::Vec3b>(rows, cols)[0] + 80;
			_reshapedFrame.at<cv::Vec3b>(rows, cols)[1] = _frame.at<cv::Vec3b>(rows, cols)[1];
			_reshapedFrame.at<cv::Vec3b>(rows, cols)[2] = _frame.at<cv::Vec3b>(rows, cols)[2];
		}
	}
	return _reshapedFrame;
}

cv::Mat DataAugmentation::Perspective()
{
	// Input Quadilateral or Image plane coordinates
	cv::Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	cv::Point2f outputQuad[4];

	// Lambda Matrix
	cv::Mat lambda(2, 4, CV_32FC1);
	//Input and Output Image;
	cv::Mat input;

	//Load the image
	input = _frame;
	// Set the lambda matrix the same type and size as input
	lambda = cv::Mat::zeros(input.rows, input.cols, input.type());

	// The 4 points that select quadilateral on the input , from top-left in clockwise order
	// These four pts are the sides of the rect box used as input 
	inputQuad[0] = cv::Point2f(-30, -60);
	inputQuad[1] = cv::Point2f(input.cols + 50, -50);
	inputQuad[2] = cv::Point2f(input.cols + 100, input.rows + 50);
	inputQuad[3] = cv::Point2f(-50, input.rows + 50);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = cv::Point2f(0, 0);
	outputQuad[1] = cv::Point2f(input.cols - 1, 0);
	outputQuad[2] = cv::Point2f(input.cols - 1, input.rows - 1);
	outputQuad[3] = cv::Point2f(0, input.rows - 1);

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	// Apply the Perspective Transform just found to the src image
	warpPerspective(input, _reshapedFrame, lambda, _reshapedFrame.size());

	return _reshapedFrame;
}
