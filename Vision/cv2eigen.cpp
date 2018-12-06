#include "cv2eigen.h"



cv2eigen::cv2eigen()
{
}


cv2eigen::~cv2eigen()
{
}

Eigen::VectorXf cv2eigen::Convert(cv::Mat imageCV)
{
	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat(reinterpret_cast<float*>(imageCV.data), imageCV.rows, imageCV.cols * imageCV.channels());
	Eigen::Map<Eigen::RowVectorXf> image(mat.data(), mat.size());

	return image;
}
