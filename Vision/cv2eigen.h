#pragma once
#include "AllIncludes.h"

class cv2eigen
{
public:
	cv2eigen();
	~cv2eigen();

	static Eigen::VectorXf Convert(cv::Mat imageCV);
};

