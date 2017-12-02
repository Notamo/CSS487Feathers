#include "FeatherBOW.h"

bool checkEqualRow(const Mat& mat1, const Mat& mat2)
{
	if (mat1.size() != mat2.size())
	{
		return false;
	}

	Mat dst;
	bitwise_xor(mat1, mat2, dst);
	if (countNonZero(dst) > 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

FeatherBOW::FeatherBOW(ExtractType eType, int numWords)
{
	extrType = eType;
	this->numWords = numWords;

	//termination criteria (figure out the details of this later
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);		//iterative, 100 iterations?
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;			//how we get the centers for iteration 1
	bowTrainer = new BOWKMeansTrainer(numWords, tc, 1, flags);

	extractor.GetExtractor(eType, FD);
	DM = DescriptorMatcher::create("FlannBased");
	bowDE = new BOWImgDescriptorExtractor(FD, DM);
}


FeatherBOW::~FeatherBOW()
{
}

//use the input images to make a dictionary of features
//for now use every image to make a dictionary
void FeatherBOW::MakeDictionary(const vector<TrainingSet> &trainingSets)
{
	//take every image and add it to a bag of words
	for (int set = 0; set < trainingSets.size(); set++)
	{
		cout << "adding \"/" << trainingSets[set].name << "\" to the BOW" << endl;

		for (int i = 0; i < trainingSets[set].images.size(); i++)
		{
			vector<KeyPoint> keypoints;
			Mat descriptors;

			//1. run the image through the feature extractor
			extractor.ExtractFeatures(extrType, trainingSets[set].images[i], keypoints, descriptors);

			//2. put the features into the BOW Trainer
			bowTrainer->add(descriptors);
		}
	}

	cout << "Clustering " << bowTrainer->descriptorsCount() << " features" << endl;

	dictionary = bowTrainer->cluster();

	cout << "Setting bowDE vocabulary" << endl;
	bowDE->setVocabulary(dictionary);

	trained = true;
}

Mat FeatherBOW::GetDictionary()
{
	return dictionary;
}

int FeatherBOW::GetSize()
{
	return numWords;
}


//Computes an image histogram, according to the current BOW
bool FeatherBOW::ComputeImgHist(const Mat &image, Mat &response_hist)
{
	if (trained == false)
	{
		cerr << "Cannot get histogram, No dictionary!" << endl;
		return false;
	}

	vector<KeyPoint> keypoints;
	Mat descriptors;
	extractor.ExtractFeatures(extrType, image, keypoints, descriptors);

	bowDE->compute(image, keypoints, response_hist);

	return true;
}

void FeatherBOW::LoadDictionary(string directory)
{
	FileStorage fs(directory + "dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	trained = true;
}

void FeatherBOW::SaveDictionary(string directory)
{
	if (!trained)
	{
		cerr << "BOW not trained! Cannot save data!" << endl;
		return;
	}

	FileStorage fs(directory + "dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}


