#include "FeatherIdentifier.h"

ExtractType ExtractTypeFromString(const string &str)
{
	if (str == "SIFT")
	{
		return ExtractType::E_SIFT;
	}
	else if (str == "SURF")
	{
		return ExtractType::E_SURF;
	}
	else if (str == "HoNC")
	{
		return ExtractType::E_HoNC;
	}
	else
	{
		return ExtractType::E_None;
	}
}

FeatherIdentifier::FeatherIdentifier(const string &workingDirectory)
{
	this->workingDirectory = workingDirectory;
}


FeatherIdentifier::~FeatherIdentifier()
{
}


bool FeatherIdentifier::Train(const string &trainingFile)
{
	ExtractType eType = ExtractType::E_None;
	int numWords = -1;

	if (!MakeTrainingSets(trainingFile, eType, numWords))
	{
		cerr << "Failed to make training sets!" << endl;
		return false;
	}

	cout << "Done Building Sets" << endl;

	cout << "Creating Vocabulary" << endl;
	if (!CreateVocabulary())
	{
		cerr << "Failed to create vocabulary!" << endl;
		return false;
	}

	cout << "Calculating Histograms" << endl;
	if (!CalculateHistograms(histograms, labels))
	{
		cerr << "Failed to create histograms!" << endl;
		return false;
	}

	cout << "Training the SVM" << endl;
	if (!TrainSVM(histograms, labels))
	{
		cerr << "Failed to train SVM!" << endl;
		return false;
	}

	trained = true;

	TestSVM();
	return true;
}

bool FeatherIdentifier::TestSVM()
{
	if (!trained)
	{
		cerr << "Cannot Identify, no training data!" << endl;
		return false;
	}

	Ptr<FeatureDetector> detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");		//combine with data from training


	BOWImgDescriptorExtractor BOWide(extractor, matcher);
	BOWide.setVocabulary(vocabulary);
	
	for (int set = 0; set < trainingSets.size(); set++)
	{
		for (int im = 0; im < trainingSets[set].images.size(); im++)
		{
			vector<KeyPoint> keypoints;
			detector->detect(trainingSets[set].images[im], keypoints);

			if (keypoints.empty())
			{
				cout << "No keypoints!" << endl;
				continue;
			}

			Mat descriptor;
			BOWide.compute(trainingSets[set].images[im], keypoints, descriptor);

			if (descriptor.empty())
			{
				cout << "No descriptor!" << endl;
				continue;
			}

			Mat results;
			float res = classifier->predict(descriptor);

			string predicted = trainingSets[(int)res].name;

			cout << "result of prediction: (" << predicted << "): " << res << endl;
			cout << "actual answer: " << trainingSets[set].name << endl;

			imshow(predicted, trainingSets[set].images[im]);
			waitKey(0);
			destroyWindow(predicted);
		}
	}

	cout << "Done Testing!" << endl;
	return true;
}
bool FeatherIdentifier::Identify(const string &testFile)
{


	return false;
}

//Read in a file <dbFile> that describes the name of each set,
//POTENTIAL FORMAT:
//<Extraction method>	<# Words>
//<set name>	<set size>	<directory>	
//<set name>	<set size>	<directory>
bool FeatherIdentifier::MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords)
{
	//try to open the trainingFile
	ifstream TF(workingDirectory + trainingFile);

	if (!TF.is_open())
	{
		cerr << "Failed to open trainingFile! (" << workingDirectory + trainingFile << ")" << endl;
		return false;
	}

	//read the first line, should be in the following format\:
	//<SIFT | SURF | HoNC>	<TAB>	<# words>
	string line;
	stringstream ss;
	if (getline(TF, line))
	{
		string methodStr;

		ss << line;
		ss >> methodStr >> numWords;

		//convert the methodString into
		eType = ExtractTypeFromString(methodStr);

		if (eType == ExtractType::E_None)
		{
			cerr << "Invalid extract type!" << endl;
			TF.close();
			return false;
		}

		if (numWords < MIN_FEATHER_WORDS)
		{
			cerr << "Invalid number of words!" << endl;
			TF.close();
			return false;
		}
	}
	else
	{
		TF.close();
		return false;
	}

	int label = 0;
	while (getline(TF, line))
	{
		ImageSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;


		if (!BuildTrainingSet(subdir, name, label, qty, TS))
		{
			cerr << "Failed to build training set for \"" << name << "\"" << endl;
			TF.close();
			return false;
		}

		trainingSets.push_back(TS);
		label++;
	}

	TF.close();
	return true;
}

bool FeatherIdentifier::BuildTrainingSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &set)
{
	cout << "Building Set: (" << name << " " << qty << " " << subdir << ")" << endl;
	set.name = name;
	set.label = label;

	for (int i = 0; i < qty; i++)
	{
		//string file(workingDirectory + subdir + name + "_" + to_string(i) + ".jpg");
		//string file(workingDirectory + subdir + name + " (" + to_string(i) + ").jpg");
		string file(workingDirectory + subdir + name + "_000" + to_string(i) + "_test.jpg");

		cout << "loading image \"" << file << "\"" << endl;
		Mat m = imread(file);

		if (m.data == nullptr)
		{
			cerr << "Could not open \"" << file << "\"!" << endl;
			cerr << "Failed to build " << name << " set!" << endl;
			return false;
		}

		set.images.push_back(m);
	}
	return true;
}


bool FeatherIdentifier::CreateVocabulary()			//further additions: <int numwordss>
{
	if (trainingSets.empty())
	{
		cerr << "training set empty!" << endl;
		return false;
	}


	//further additions: change algorithm
	Ptr<FeatureDetector> detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();

	//make the correct descriptors material for this extractor
	Mat trainingDescriptors(1, extractor->descriptorSize(), extractor->descriptorType());

	vocabulary.create(0, 1, CV_32FC1);

	for (int set = 0; set < trainingSets.size(); set++)
	{
		for (int im = 0; im < trainingSets[set].images.size(); im++)
		{

			//first get the keypoints
			vector<KeyPoint> keypoints;
			detector->detect(trainingSets[set].images[im], keypoints);

			if (!keypoints.empty())
			{
				Mat descriptors;
				extractor->compute(trainingSets[set].images[im], keypoints, descriptors);

				if (!descriptors.empty())
				{
					//add he descriptor to the set of trainingDescriptors
					trainingDescriptors.push_back(descriptors);
				}
			}
		}
	}

	if (trainingDescriptors.empty())
	{
		cerr << "No training descriptors were fournd!" << endl;
		return false;
	}

	cout << "Clustering...";

	//do the actual training to get a vocab
	BOWKMeansTrainer trainer(1000);
	trainer.add(trainingDescriptors);
	vocabulary = trainer.cluster();

	cout << "Done!" << endl;
	return true;
}


bool FeatherIdentifier::CalculateHistograms(Mat &outSamples, Mat &outLabels)
{
	if (trainingSets.empty())
	{
		cerr << "training set empty!" << endl;
		return false;
	}

	if (vocabulary.empty())
	{
		cerr << "Vocabulary is empty!" << endl;
		return false;
	}

	Ptr<FeatureDetector> detector = SIFT::create();
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");	//we should be able to change these methods

	BOWImgDescriptorExtractor BOWIde(extractor, matcher);
	BOWIde.setVocabulary(vocabulary);

	Mat samples;
	outSamples.create(0, 1, CV_32FC1);
	outLabels.create(0, 1, CV_32SC1);

	for (int set = 0; set < trainingSets.size(); set++)
	{
		for (int im = 0; im < trainingSets[set].images.size(); im++)
		{
			vector<KeyPoint> keypoints;
			detector->detect(trainingSets[set].images[im], keypoints);

			if (!keypoints.empty())
			{
				Mat descriptors;
				BOWIde.compute(trainingSets[set].images[im], keypoints, descriptors);

				if (!descriptors.empty())
				{
					if (samples.empty())
					{
						samples.create(0, descriptors.cols, descriptors.type());
					}
					
					cout << "adding " << descriptors.rows << " " << trainingSets[set].name << " sample" << endl;
					samples.push_back(descriptors);

					Mat classLabels(descriptors.rows, 1, CV_32SC1);

					for (int r = 0; r < classLabels.rows; r++)
					{
						for (int c = 0; c < classLabels.cols; c++)
						{
							classLabels.at<int>(r, c) = trainingSets[set].label;
						}
					}

					outLabels.push_back(classLabels);

				}
			}
		}
	}

	if (samples.empty() || outLabels.empty())
	{
		cerr << "samples are empty!" << endl;
		return false;
	}

	samples.convertTo(outSamples, CV_32FC1);
	return true;
}

bool FeatherIdentifier::TrainSVM(const Mat &samples, const Mat &labels)
{
	if (samples.empty() || samples.type() != CV_32FC1)
	{
		cerr << "Bad samples!" << endl;
		return false;
	}
	else if (labels.empty() || labels.type() != CV_32SC1)
	{
		cerr << "Bad labels!" << endl;
		return false;
	}

	classifier = SVM::create();

	return classifier->train(samples, ROW_SAMPLE, labels);
}