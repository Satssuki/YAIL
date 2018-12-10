#pragma once
#include "AllIncludes.h"

class EigenSerializer
{
public:
	EigenSerializer();
	~EigenSerializer();

	static void SaveMatrix(std::ofstream& of, Eigen::MatrixXf mat);
	static void SaveVector(std::ofstream& of, Eigen::VectorXf vec);

	static void LoadMatrix(std::ifstream& inf, Eigen::MatrixXf& mat);
	static void LoadVector(std::ifstream& inf, Eigen::VectorXf& vec);
};

