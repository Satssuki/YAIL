#pragma once
#include "AllIncludes.h"

class Guess
{
public:
	Guess();
	~Guess();

	void StartTest(std::vector<cv::Mat>* images, std::vector<int>* labels);
	static void Print(int label, int result, cv::Mat image);

private:

};

