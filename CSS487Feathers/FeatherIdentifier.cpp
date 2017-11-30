#include "FeatherIdentifier.h"



FeatherIdentifier::FeatherIdentifier(FIDMode mode)
{
	mMode = mode;
}


FeatherIdentifier::~FeatherIdentifier()
{
}

void FeatherIdentifier::Run(const vector<Mat> &trainingData, const Mat &testData)
{
	switch (mMode)
	{
		case Train_And_ID:
		{
			TrainBOWs();
			Identify(testData);
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
			Identify(testData);
			break;
		}
	}
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

void FeatherIdentifier::ListResults(const Mat& testImg, vector<RatingPair> pairs)
{

}