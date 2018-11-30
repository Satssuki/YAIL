#pragma once
#include "AllIncludes.h"

class DataLoader
{
public:
	DataLoader();
	~DataLoader();
	  
	std::tuple < std::vector<cv::Mat>, std::vector<int>> LoadData(std::string path);

private:
	unsigned Seed;
	cv::Size Size = cv::Size(64, 64);
	std::string ClassNames[2] = { "cat", "dog" }; // 0 = cat, 1 = dog
};

