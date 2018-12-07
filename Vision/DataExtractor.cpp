#include "DataExtractor.h"

std::vector<cv::Mat> DataExtractor::ExtractCharacters(cv::Mat & input) {
	cv::Mat temp, gray;
	cv::cvtColor(input, gray, CV_BGR2GRAY);
	gray = gray > 127;

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	cv::Rect minRect;
}
