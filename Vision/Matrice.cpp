#include "Matrice.h"


//Constructor
Matrice::Matrice(Mat &src):
	r(Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>>(src.ptr<float>(), src.rows, src.cols)),
	g(Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>>(src.ptr<float>() + 1, src.rows, src.cols)),
	b(Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, RowMajor, Stride<3, 1>>(src.ptr<float>() + 2, src.rows, src.cols))
{

}


Matrice::~Matrice(){}

void Matrice::Transpose()
{
	r.transpose();
	g.transpose();
	b.transpose();
}

void Matrice::Resize(int rows, int cols)
{
	r.resize(rows, cols);
	g.resize(rows, cols);
	b.resize(rows, cols);
}
