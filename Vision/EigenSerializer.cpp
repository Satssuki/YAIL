#include "EigenSerializer.h"



EigenSerializer::EigenSerializer()
{

}


EigenSerializer::~EigenSerializer()
{
}

void EigenSerializer::SaveMatrix(std::ofstream& of, Eigen::MatrixXf mat)
{
	int rows = mat.rows();
	of.write((char*)&rows, sizeof(int));

	int cols = mat.cols();
	of.write((char*)&cols, sizeof(int));

	float* data = mat.data();
	of.write((char*)&data[0], sizeof(float) * rows * cols);
}

void EigenSerializer::SaveVector(std::ofstream& of, Eigen::VectorXf vec)
{
	int rows = vec.rows();
	of.write((char*)&rows, sizeof(int));

	float* data = vec.data();
	of.write((char*)&data[0], sizeof(float) * rows);
}

void EigenSerializer::LoadMatrix(std::ifstream& inf, Eigen::MatrixXf & mat)
{
	int rows = mat.rows();
	inf.read((char*)&rows, sizeof(int));

	int cols = mat.cols();
	inf.read((char*)&cols, sizeof(int));

	float* data = mat.data();
	inf.read((char*)&data[0], sizeof(float)* rows * cols);
}

void EigenSerializer::LoadVector(std::ifstream& inf, Eigen::VectorXf & vec)
{
	int rows = vec.rows();
	inf.read((char*)&rows, sizeof(int));

	float* data = vec.data();
	inf.read((char*)&data[0], sizeof(float)* rows);
}

