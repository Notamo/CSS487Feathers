#include "FeatherIdentifier.h"

#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	//Mat img = imread("test.jpg");

	if (!img.data)
	{
		cout << "Could not open test.jpg!" << endl;
		return -1;
	}

	FeatherIdentifier FeatherID = FeatherIdentifier(FIDMode::Train_And_ID);

		//I saw this in prof's code, it might help
		//KeyPointsFilter::retainBest(keypoints, 1000);



		namedWindow("keypoints");
		imshow("keypoints", img_keypoints);
		waitKey(0);
	}

	
	return 0;
}