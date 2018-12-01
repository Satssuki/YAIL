#pragma once
#include "AllIncludes.h"

#define ROWS 64
#define COLS 64

using namespace cv;
using namespace Eigen;

class Matrice
{
public:
	//SIZE OF 64x64 only;
	Matrice(Mat &src);
	~Matrice();


	//RGB
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>> r;
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>> g;
	Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>> b;

	void Transpose();
	void Resize(int rows, int cols);
};

