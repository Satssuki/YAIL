#pragma once
#include "AllIncludes.h"

class DataLoader
{
public:
	DataLoader();
	~DataLoader();
	  
	std::tuple < std::vector<cv::Mat>, std::vector<int>> LoadData(std::string rootPath);

	std::string ClassNames[2] = { "cat", "dog" }; // 0 = cat, 1 = dog
private:
	void LoadImages(std::string path, std::vector<cv::Mat>* images);

	unsigned Seed;
	cv::Size Size = cv::Size(64, 64);
};

