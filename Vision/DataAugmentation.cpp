#include "DataAugmentation.h"

// Les translations et blur sont trop intense. Elle doit etre fait selon la dimension de l'image et non hardcodé
void DataAugmentation::GetAugmentedFrame(cv::Mat &input, bool flip)
{
	AugmentationType _transformation = flip ? (AugmentationType)(rand() % FLIPPING + 1) : (AugmentationType)(rand() % PERSPECTIVE + 1);
	switch (_transformation) {
	case (DISTORTION):
		Distortion(input);
		break;
	case (TRANSLATION):
		Translate(input);
		break;
	case (ROTATION):
		Rotate(input);
		break;
	case (FLIPPING):
		Flip(input);
		break;
	case (PEPPER_AND_SALT):
		SaltNPepper(input);
		break;
	case (NOISE):
		Noise(input);
		break;
	case (LIGHTNING):
		Lightning(input);
		break;
	case (PERSPECTIVE):
		Perspective(input);
		break;
	default:
		return;
	}

} 

void DataAugmentation::Distortion(cv::Mat &input)
{
	double sigma = 0.75;
	double alpha = 0.5;
	bool bNorm = false;

	cv::Mat dx(input.size(), CV_64FC1);
	cv::Mat dy(input.size(), CV_64FC1);

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
	if (bNorm) {
		dx /= cv::norm(dx, cv::NORM_L1);
		dy /= cv::norm(dy, cv::NORM_L1);
	}

	//The displacement fields are then multiplied by a scaling factor alpha
	//that controls the intensity of the deformation.
	dx *= alpha;
	dy *= alpha;

	//Inverse(or Backward) Mapping to avoid gaps and overlaps.
	cv::Rect checkError(0, 0, input.cols, input.rows);
	int nCh = input.channels();

	for (int displaced_y = 0; displaced_y < input.rows; displaced_y++) {
		for (int displaced_x = 0; displaced_x < input.cols; displaced_x++)
		{
			int org_x = displaced_x - dx.at<double>(displaced_y, displaced_x);
			int org_y = displaced_y - dy.at<double>(displaced_y, displaced_x);

			if (checkError.contains(cv::Point(org_x, org_y)))
			{
				for (int ch = 0; ch < nCh; ch++)
				{
					input.data[(displaced_y * input.cols + displaced_x) * nCh + ch] = input.data[(org_y * input.cols + org_x) * nCh + ch];
				}
			}
		}
	}
}

void DataAugmentation::Translate(cv::Mat &input)
{
	float offsetX = rand() % input.rows*0.05 - (input.rows*0.05/2);
	float offsetY = rand() % input.cols*0.05 - (input.cols*0.05 / 2);
	cv::Mat trans_mat = (cv::Mat_<double>(2, 3) << 1, 0, offsetX, 0, 1, offsetY);
	cv::warpAffine(input, input, trans_mat, input.size());
}

void DataAugmentation::Rotate(cv::Mat &input)
{
    cv:: Point2f src_center(input.cols / 2.0F, input.rows / 2.0F);
	cv::Mat rot_mat = getRotationMatrix2D(src_center, rand() % 30 - 15, 1.0);
	cv::warpAffine(input, input, rot_mat, input.size());
}

void DataAugmentation::Flip(cv::Mat &input)
{
	cv::flip(input, input, 1);
}

void DataAugmentation::SaltNPepper(cv::Mat &input)
{
	cv::Mat saltPepperNoise = cv::Mat::zeros(input.rows, input.cols, CV_8U);
	cv::randu(saltPepperNoise, 0, 255);

	cv::Mat black = saltPepperNoise < 15;
	cv::Mat white = saltPepperNoise > 245;

	input.setTo(255, white);
	input.setTo(0, black);
}

void DataAugmentation::Noise(cv::Mat &input)
{
	cv::blur(input, input, cv::Size(5, 5));
}

void DataAugmentation::Lightning(cv::Mat &input)
{
	input.convertTo(input, 1, 40, 40);
}

void DataAugmentation::Perspective(cv::Mat &input)
{
	// Input Quadilateral or Image plane coordinates
	cv::Point2f inputQuad[4];
	// Output Quadilateral or World plane coordinates
	cv::Point2f outputQuad[4];

	// Lambda Matrix
	cv::Mat lambda(2, 4, CV_32FC1);

	// Set the lambda matrix the same type and size as input
	lambda = cv::Mat::zeros(input.rows, input.cols, input.type());

	// The 4 points that select quadilateral on the input , from top-left in clockwise order
	// These four pts are the sides of the rect box used as input 
	inputQuad[0] = cv::Point2f(-7, -15);
	inputQuad[1] = cv::Point2f(input.cols + 12, -12);
	inputQuad[2] = cv::Point2f(input.cols + 25, input.rows + 12);
	inputQuad[3] = cv::Point2f(-12, input.rows + 12);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = cv::Point2f(0, 0);
	outputQuad[1] = cv::Point2f(input.cols - 1, 0);
	outputQuad[2] = cv::Point2f(input.cols - 1, input.rows - 1);
	outputQuad[3] = cv::Point2f(0, input.rows - 1);

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	// Apply the Perspective Transform just found to the src image
	warpPerspective(input, input, lambda, input.size());
}
