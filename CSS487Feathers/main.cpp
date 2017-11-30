#include "FeatherIdentifier.h"

#include <iostream>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

int main(int argc, char *argv[])
{
	string dbFile, inputFile;
	inputFile = argv[1];
	dbFile = argv[2];

	FeatherIdentifier FeatherID = FeatherIdentifier(FIDMode::Train_And_ID);

	if (!FeatherID.Run(dbFile, inputFile))
		return -1;

	//I saw this in prof's code, it might help
	//KeyPointsFilter::retainBest(keypoints, 1000);
	return 0;
}