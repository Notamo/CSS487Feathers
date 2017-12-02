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

#include "FeatherIDUtil.h"
#include "FeatherBOW.h"


class FeatherIdentifier
{
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

	vector<TrainingSet> trainingSets;
	Ptr<FeatherBOW> BOW;
	Ptr<BOWImgDescriptorExtractor> bowDE;
	vector<Ptr<SVM>> SVMs;

	//Core Functionality
	bool TrainBOWs(ExtractType eType, int numWords);
	bool TrainSVMs(ExtractType eType, int numWords);
	bool TrainSVM(int index, int numWords);
	bool MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords);
	bool BuildTrainingSet(const string &directory, const string &prefix, const int &qty, TrainingSet &set);
};

