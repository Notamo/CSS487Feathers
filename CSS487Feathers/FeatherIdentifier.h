#pragma once
#include <string>
#include <vector>
using namespace std;

#include <opencv2\core.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace cv;


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
private:
	//Utility (maybe put these structs outside the class?)
	typedef struct
	{
		vector<Mat> images;
		string name;
	} TrainingSet;

	typedef struct
	{
		FeatherBOW *BOW = nullptr;
		float probability = 0.0f;
	} RatingPair;

public:
	FeatherIdentifier(FIDMode mode);
	~FeatherIdentifier();

	bool Run(const string &dbFile, const string &inputFile);

private:
	FIDMode mMode = FIDMode::None;
	vector<FeatherBOW> BOWs;

	//Core Functionality
	void TrainBOWs();
	void SaveBOWs();
	void LoadBOWs();

	void Identify(const Mat &testData);

	//UI Assistance (display results of test)
	void ListResults(const Mat &testImg, const vector<RatingPair> &pairs);




	bool MakeTrainingSets(const string &dbFile, vector<TrainingSet> &sets);

};

