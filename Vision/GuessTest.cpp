#include "GuessTest.h"



GuessTest::GuessTest()
{
	
}


GuessTest::~GuessTest()
{
}

void GuessTest::StartTest(std::vector<cv::Mat>* images, std::vector<int>* labels)
{
	std::cout << "Try to guess if the image contains a cat (press 0) or a dog (press 1)." << std::endl;
	std::cout << "Enter the number of images you want to test" << std::endl;
	
	int totalImages;
	std::cin >> totalImages;
	if (totalImages < 0 || totalImages > images->size()) 
	{
		std::cout << "Error number images" << std::endl;
		return;
	}

	// the test is easy because the images are resized
	int goodAnswer = 0;
	for (int i = 0; i < totalImages; i++)
	{
		cv::Mat out = (*images)[i];
		resize((*images)[i], out, cv::Size(200, 200));
		cv::imshow("Guess Test", out);

		char key = cv::waitKey(0);
		char answer = (char)((*labels)[i] + 48);
		if (key == answer)
		{
			goodAnswer++;
			std::cout << "Good answer!" << std::endl;
		} else { std::cout << "Wrong answer!" << std::endl; }
	}

	std::cout << "You had " << goodAnswer << "/" << totalImages << " good answers!" << std::endl;
}
