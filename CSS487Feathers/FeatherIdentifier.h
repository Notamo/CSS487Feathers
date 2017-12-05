#pragma once
#include <string>
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

class FeatherIdentifier
{
public:
	FeatherIdentifier(const string &workingDirectory);
	~FeatherIdentifier();

	bool Train(const string &trainingFile);
	bool Identify(const string &testFile);

private:
	string workingDirectory;

	//properties of the training state (cannot be cont
	FeatureExtractor Extractor;
	ExtractType eType;
	int numWords;

	Mat vocabulary;
	Mat histograms;
	Mat labels;
	Ptr<SVM> classifier;
	bool trained = false;

	vector<ImageSet> trainingSets;
	vector<ImageSet> testingSets;

	bool RunTestingSets(const vector<ImageSet> &testingSets);

	bool MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords);
	bool BuildTrainingSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &trainingSet);

	bool MakeTestingSets(const string &trainingFile);
	bool BuildTestingSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &testingSet);


	bool CreateVocabulary(Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE);
	bool CalculateHistograms(Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE, Mat &outSamples, Mat &outLabels);
	bool TrainSVM(const Mat &samples, const Mat &labels);

	bool TestSVM(ExtractType eType, Ptr<FeatureDetector> &FD, Ptr<DescriptorExtractor> &DE, vector<ImageSet> &sets);
};

