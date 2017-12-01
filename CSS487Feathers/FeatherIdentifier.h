#pragma once
#include <string>
#include <strstream>
#include <vector>
#include <fstream>
using namespace std;

#include <opencv2\core.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
using namespace cv;

#include "FeatherBOW.h"

typedef struct
{
	string name;
	float rating = 0.0f;
} RatingPair;


class FeatherIdentifier
{
private:

	typedef struct
	{
		vector<Mat> images;
		Ptr<FeatherBOW> BOW;
		string name;
	} TrainingSet;

public:
	FeatherIdentifier(const string &workingDirectory);
	~FeatherIdentifier();

	bool TrainIdentifier(const string &trainingFile);
	bool Identify(const string &testFile, vector<RatingPair> &ratings);

	bool SaveBOWs(const string &bowFile);
	bool LoadBOWs(const string &bowFile);

	//UI Assistance (display results of test)
	void ListResults(const vector<RatingPair> &pairs);

private:
	string workingDirectory;
	bool trained = false;
	ExtractType eType = ExtractType::E_None;

	//TRAINING SETS VECTOR GOES HERE
	vector<TrainingSet> trainingSets;
	vector<Ptr<FeatherBOW>> BOWs;
	//The SVM (Support vector machine) classifies and predicts for us
	Ptr<SVM> svm;

	//Core Functionality
	bool TrainBOWs(ExtractType eType, int numWords);
	bool TrainSVM(ExtractType eType, int numWords);
	bool MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords);
	bool BuildTrainingSet(const string &directory, const string &prefix, const int &qty, TrainingSet &set);
};

