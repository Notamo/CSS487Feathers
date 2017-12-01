#pragma once
#include <opencv2\features2d.hpp>
#include <opencv2\ml.hpp>
using namespace cv;
using namespace ml;

#include <string>
#include <vector>
using namespace std;

#include "FeatureExtractor.h"

const int MIN_FEATHER_WORDS = 10;

/*
FeatherBOW
Description:
This class Represents a Bag of Words (BOW) for a specific feather
The BOW can either be trained from a Data Set, or the already trained
Data can be loaded in from a file. Once trained data is in the BOW,
An input can be compared to the BOW to determine how likely it is in the set
*/
class FeatherBOW
{
public:
	FeatherBOW(ExtractType eType, int numWords, string name);
	~FeatherBOW();

	//Creates the BOW, by training a set of Images
	void Train(const vector<Mat> &inputImages);

	//Tells the user how likely it is the input image falls under this category
	float Predict(const Mat &input);

	string GetName() { return name; }
	//STRETCH GOAL: load and save training state for faster detection
	void LoadData();
	void SaveData();
private:
	ExtractType extrType = ExtractType::E_None;
	int numWords = 10;
	string name = "[no name]";
	bool trained = false;
	
	
	//for generating a bag of words/histogram (one for each training image)
	Ptr<BOWKMeansTrainer> bowTrainer;
	int numAttempts = 3;

	//Support Vector Machine actually generates a way to predict
	//using our GIANT histograms
	Ptr<SVM> svm;

	
};

