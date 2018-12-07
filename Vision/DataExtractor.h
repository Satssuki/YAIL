#pragma once
#include "AllIncludes.h"
class DataExtractor
{
public:
	static std::vector<cv::Mat> ExtractCharacters(cv::Mat &input);
private:
	std::vector<cv::Mat> characters;
};

