#pragma once
#include <string>
#include <iostream>
#include <strstream>
#include <vector>
#include <fstream>
using namespace std;

#include <opencv2\core.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

#include "FeatherIDUtil.h"
#include "FeatureExtractor.h"
#include "BOWDataMgmt.h"

/*
FeatherIdentifier
A Class for Identifying Feathers using
Bag of Words and a Scaled Vector Machine
*/
class FeatherIdentifier : public BOWDataMgmt
{
public:
	FeatherIdentifier(const string &workingDirectory);
	~FeatherIdentifier();

	bool Train(const string &trainingFile, bool verify, bool verbose);
	bool Identify(const string &testFile, bool verbose);

	bool Save(const string &saveName);
	bool Load(const string &loadName);

private:
	//properties of the training state
	FeatureExtractor FExtractor;
	ExtractType eType;
	int numWords;

	Mat vocabulary;
	Mat histograms;
	Mat labels;
	Ptr<SVM> classifier;

	bool trained = false;

	vector<ImageSet> trainingSets;
	vector<ImageSet> testingSets;

	bool CreateDictionary(Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE);
	bool CalculateHistograms(Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE, Mat &outSamples, Mat &outLabels);
	bool TrainSVM(const Mat &samples, const Mat &labels);
	bool TestSVM(ExtractType eType, Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE, vector<ImageSet> &trainingSets, vector<ImageSet> &testingSets, bool verbose);
};

