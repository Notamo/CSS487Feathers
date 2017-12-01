#include "FeatherBOW.h"



FeatherBOW::FeatherBOW(ExtractType eType, int numWords, string name)
{
	extrType = eType;
	this->numWords = numWords;
	this->name = name;

	//termination criteria (figure out the details of this later
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);		//iterative, 100 iterations?
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;			//how we get the centers for iteration 1
	bowTrainer = new BOWKMeansTrainer(numWords, tc, 1, flags);
}


FeatherBOW::~FeatherBOW()
{
}

//use the input images to make a dictionary of features
//for now use every image to make a dictionary
void FeatherBOW::MakeDictionary(const vector<Mat> &inputImages)
{
	//3. Make the BOW dictionary
	//<for each entry in the training data>					//apparently we don't usually use every image to make the dictionary
	for (int i = 0; i < inputImages.size(); i++)
	{
		vector<KeyPoint> keypoints;
		Mat descriptors;

		//1. run the data through the feature extractor
		extractor.ExtractFeatures(extrType, inputImages[i], keypoints, descriptors);

		cout << "keypoints: " << keypoints.size() << endl;
		cout << "descriptors.size(): " << descriptors.size() << endl;

		//2. put the features into the BOW Trainer
		bowTrainer->add(descriptors);
	}

	cout << "Clustering " << bowTrainer->descriptorsCount() << " features" << endl;

	dictionary = bowTrainer->cluster();

	cout << "dictionary size: " << dictionary.size() << endl;
	namedWindow("dictionary");
	imshow("dictionary", dictionary);
	waitKey(0);
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
bool FeatherBOW::ComputeImgHist(const Mat &descriptors, Mat &hist)
{
	//hist = Mat(dictionary.rows, 1, CV_32FC1);
	//hist = Mat(1, dictionary.rows, CV_32FC1);
	hist = Mat_<float>(1, dictionary.rows);

	for (int dicRow = 0; dicRow < dictionary.rows; dicRow++)
	{
		for (int descRow = 0; descRow < descriptors.rows; descRow++)
		{
			//check if the rows are equal
			Mat diff;
			compare(descriptors.row(descRow), dictionary.row(dicRow), diff, CMP_NE);
			int nz = countNonZero(diff);
			if (nz == 0)
				hist.at<float>(dicRow) += 1.0f;
		}
	}

	return true;
}

void FeatherBOW::LoadData(string directory)
{
	FileStorage fs(directory + name + "dictionary.yml", FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	trained = true;
}

void FeatherBOW::SaveData(string directory)
{
	if (!trained)
	{
		cerr << name << "BOW not trained! Cannot save data!" << endl;
		return;
	}

	FileStorage fs(directory + name + "dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}
