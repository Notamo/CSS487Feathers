#include "FeatherIdentifier.h"

#include <iostream>
using namespace std;

#include <opencv2\core.hpp>
using namespace cv;

int main(int argc, char *argv[])
{
	string inputFile, trainingFile, workingDirectory;

	if (argc < 3)
		return -1;

	if (argc >= 3)
	{
		inputFile = argv[1];
		trainingFile = argv[2];
	}
	
	if (argc >= 4)
	{
		workingDirectory = argv[3];
		workingDirectory += "/";
	}
	else
	{
		workingDirectory = "";
	}

	FeatherIdentifier FeatherID = FeatherIdentifier(workingDirectory);

	if (!FeatherID.TrainIdentifier(trainingFile))
		return -1;

	vector<RatingPair> ratings;
	if (!FeatherID.Identify(inputFile, ratings))
		return -1;

	//FeatherID.ListResults(ratings);

	cin.get();

	return 0;
}