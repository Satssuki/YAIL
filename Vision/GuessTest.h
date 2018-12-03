#pragma once
#include "AllIncludes.h"

class GuessTest
{
public:
	GuessTest();
	~GuessTest();

	void StartTest(std::vector<cv::Mat>* images, std::vector<int>* labels);
};

