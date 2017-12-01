#include "FeatherBOW.h"



FeatherBOW::FeatherBOW(ExtractType eType, int numWords, string name)
{
	extrType = eType;
	this->numWords = numWords;
	this->name = name;

	bowTrainer = new BOWKMeansTrainer(numWords);
}


FeatherBOW::~FeatherBOW()
{
}

void FeatherBOW::Train(const vector<Mat> &inputImages)
{
	FeatureExtractor extractor;
	
	//<for each entry in the training data>
	for (int i = 0; i < inputImages.size(); i++)
	{
		vector<KeyPoint> keypoints;
		Mat descriptors;

		//1. run the data through the feature extractor
		extractor.ExtractFeatures(extrType, inputImages[i], keypoints, descriptors);

		namedWindow("image");
		imshow("image", inputImages[i]);
		waitKey(0);


		Mat keypointImg;
		drawKeypoints(inputImages[i], keypoints, keypointImg);
		namedWindow("keypoints");
		imshow("keypoints", keypointImg);
		waitKey(0);


		namedWindow("descriptor " + to_string(i));
		imshow("descriptor " + to_string(i), descriptors);
		waitKey(0);

		cout << descriptors.size() << endl;
		//2. put the features into the BOW Trainer
		bowTrainer->add(descriptors);
	}

	//3. Make the BOW Histogram (dictionary)
	Mat bowHistogram;
	bowHistogram = bowTrainer->cluster();

	namedWindow("histogram");
	imshow("histogram", bowHistogram);
	waitKey(0);

	//3. Make the BOW histrogram
	//BOWImgDescriptorExtractor bowDE(descriptors, mat)

	//4. Add it to the SVM	
}

void FeatherBOW::LoadData()
{

}

void FeatherBOW::SaveData()
{

}

float FeatherBOW::Predict(const Mat &input)
{
	//1. run the input throught the feature extractor
	//2. put the features through the BOW machine
	//3. SVM.predict()
	return 0.0f;
}
