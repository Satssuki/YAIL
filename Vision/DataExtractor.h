#pragma once
#include "AllIncludes.h"
class DataExtractor
{
public:
	static std::vector<cv::Mat> ExtractCharacters(cv::Mat &input);
	static void StepCharacter(cv::Mat &input);
private:
	static std::vector<cv::Mat> characters;
};

