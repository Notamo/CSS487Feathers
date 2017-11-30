#include "FeatherIdentifier.h"



FeatherIdentifier::FeatherIdentifier(FIDMode mode)
{
	mMode = mode;
}


FeatherIdentifier::~FeatherIdentifier()
{
}

bool FeatherIdentifier::Run(const string &dbFile, const string &inputFile)
{
	Mat img = imread(inputFile);	//this can fail! put in a check

	switch (mMode)
	{
		case Train_And_ID:
		{
			TrainBOWs();
			Identify(img);
			break;
		}
		case Train_And_Save:
		{
			TrainBOWs();
			SaveBOWs();
			break;
		}
		case Load_And_ID:
		{
			LoadBOWs();
			Identify(img);
			break;
		}
	}

	return true;
}


void FeatherIdentifier::TrainBOWs()
{

}

void FeatherIdentifier::SaveBOWs()
{

}

void FeatherIdentifier::LoadBOWs()
{

}

void FeatherIdentifier::Identify(const Mat &testData)
{

}

void FeatherIdentifier::ListResults(const Mat& testImg, const vector<RatingPair> &pairs)
{



	//Finally, show the image the user used to test
	namedWindow("TestImg");
	imshow("TestImg", testImg);
	waitKey(0);
}

bool FeatherIdentifier::MakeTrainingSets(const string &dbFile, vector<TrainingSet> &sets)
{
	return false;
}