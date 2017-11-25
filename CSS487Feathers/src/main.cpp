#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::xfeatures2d;

#include "Feature Detection\HoNC.h"

#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	Mat img = imread("test.jpg");

	if (!img.data)
	{
		cout << "Could not open test.jpg!" << endl;
		return -1;
	}

	//Test use of SURF, SIFT, HoNC	
	vector<KeyPoint> keypoints;
	for (int i = 0; i < 3; i++)
	{
		if (i == 0)
		{
			Ptr<SURF> surf = SURF::create();
			surf->detect(img, keypoints);
		}
		else if (i == 1)
		{
			Ptr<SIFT> sift = SIFT::create();
			sift->detect(img, keypoints);
		}
		else if (i == 2)
		{
			Ptr<HoNC> honc = HoNC::create();
			(*honc)(img, noArray(), keypoints, noArray(), false);			//pulled from prof olson's code, need to examine more
		}

		//I saw this in prof's code, it might help
		KeyPointsFilter::retainBest(keypoints, 1000);

		//draw keypoints
		Mat img_keypoints;
		drawKeypoints(img, keypoints, img_keypoints);

		namedWindow("keypoints");
		imshow("keypoints", img_keypoints);
		waitKey(0);
	}

	
	return 0;
}