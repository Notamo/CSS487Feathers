#pragma once

#include <string>
#include <strstream>
#include <iostream>
#include <fstream>
using namespace std;

#include <opencv2\core.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace cv::ml;


#include "FeatherIDUtil.h"
#include "FeatureExtractor.h"

const string saveDir = "SAVED/";

/*
BOWDataMgmt
A class to help separate the more data building
and file I/O related functions from the reso of
the program
*/
class BOWDataMgmt
{
public:
	BOWDataMgmt(const string &workingDirectory);
	~BOWDataMgmt();


protected:
	bool MakeTrainingSets(const string &trainingFile, vector<ImageSet> &trainingSets, ExtractType &eType, int &numWords);
	bool MakeTestingSets(const string &trainingFile, vector<ImageSet> &testingSets);

	bool SaveSVM(const string &fileName, Ptr<SVM> &svm);
	bool LoadSVM(const string &fileName, Ptr<SVM> &svm);

	bool SaveDictionary(const string &fileName, const Mat &dictionary);
	bool LoadDictionary(const string &fileName, Mat &dictionary);

private:
	string workingDirectory;

	//the locations where this data is/can be stored
	string vocabFile;
	string histogramFile;
	string SVMFile;

	bool BuildImageSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &testingSet);

};

