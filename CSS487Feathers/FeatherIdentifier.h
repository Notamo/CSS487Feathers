#pragma once
#include <opencv2\features2d.hpp>
using namespace cv;

#include <string>
#include <vector>
using namespace std;

#include "FeatherBOW.h"

typedef enum
{
	None,
	Train_And_ID,		//Start by just using this mode
	Train_And_Save,
	Load_And_ID
} FIDMode;

class FeatherIdentifier
{
public:
	FeatherIdentifier(FIDMode mode);
	~FeatherIdentifier();

	void Run(const vector<Mat> &trainingData, const Mat &testData);

private:
	FIDMode mMode = None;
	vector<FeatherBOW> BOWs;

	void TrainBOWs();
	void SaveBOWs();
	void LoadBOWs();

	void Identify(const Mat &testData);


	//UI Assistance (display results of test)
	typedef struct
	{
		FeatherBOW *BOW = nullptr;
		float probability = 0.0f;
	} RatingPair;

	void ListResults(const Mat &testImg, vector<RatingPair> pairs);

};

