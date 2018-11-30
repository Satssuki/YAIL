#pragma once
#include "AllIncludes.h"

class DataLoader
{
public:
	DataLoader();
	~DataLoader();
	
	std::tuple < std::vector<cv::Mat*>, std::vector<uint8_t>> LoadTestData();
	std::tuple < std::vector<cv::Mat*>, std::vector<uint8_t>> LoadTrainData();

private:
	unsigned seed;
};

