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
	//Utility (maybe put these structs outside the class?)
	typedef struct
	{
		vector<Mat> images;
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

	vector<Ptr<FeatherBOW>> BOWs;

	//Core Functionality
	bool TrainBOWs(const vector<TrainingSet> &trainingSets, ExtractType eType, int numWords);
	bool MakeTrainingSets(const string &trainingFile, vector<TrainingSet> &trainingSets, ExtractType &eType, int &numWords);
	bool BuildTrainingSet(const string &directory, const string &prefix, const int &qty, TrainingSet &set);
};

