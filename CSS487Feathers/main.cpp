#include "FeatherIdentifier.h"

#include <iostream>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

int main(int argc, char *argv[])
{
	string inputFile, trainingFile;
	inputFile = argv[1];
	trainingFile = argv[2];

	FeatherIdentifier FeatherID = FeatherIdentifier();

	FeatherID.TrainIdentifier(trainingFile);

	vector<RatingPair> ratings;
	FeatherID.Identify(inputFile, ratings);

	FeatherID.ListResults(ratings);

	//I saw this in prof's code, it might help
	//KeyPointsFilter::retainBest(keypoints, 1000);
	return 0;
}