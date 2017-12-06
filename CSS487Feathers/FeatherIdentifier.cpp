#include "FeatherIdentifier.h"



FeatherIdentifier::FeatherIdentifier(const string &workingDirectory)
{
	this->workingDirectory = workingDirectory;
}


FeatherIdentifier::~FeatherIdentifier()
{
}


bool FeatherIdentifier::Train(const string &trainingFile, bool verify, bool verbose)
{
	eType = ExtractType::E_None;
	numWords = -1;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;

	cout << "Making Training Sets" << endl;
	if (!MakeTrainingSets(trainingFile, eType, numWords))
	{
		cerr << "Failed to make training sets!" << endl;
		return false;
	}
	FExtractor.GetDE(eType, detector);
	FExtractor.GetFD(eType, extractor);

	cout << endl << "Creating Vocabulary" << endl;
	if (!CreateVocabulary(detector, extractor))
	{
		cerr << "Failed to create vocabulary!" << endl;
		return false;
	}

	cout << endl << "Calculating Histograms" << endl;
	if (!CalculateHistograms(detector, extractor, histograms, labels))
	{
		cerr << "Failed to create histograms!" << endl;
		return false;
	}

	cout << endl << "Training the SVM" << endl;
	if (!TrainSVM(histograms, labels))
	{
		cerr << "Failed to train SVM!" << endl;
		return false;
	}


	cout << "Training Complete!" << endl;
	trained = true;

	cout << endl << "Testing SVM with training data" << endl << endl;

	if(verify)
		TestSVM(eType, detector, extractor, trainingSets, trainingSets, false, true);		//test the training set against itself

	return true;
}

bool FeatherIdentifier::Identify(const string &testingFile, bool showImg, bool verbose)
{
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;

	cout << endl << "Identifying!" << endl; 
	if (!trained)
	{
		cerr << "Cannot Identify, no training data!" << endl;
		return false;
	}

	if (!MakeTestingSets(testingFile))
	{
		cerr << "Failed to make testing set!" << endl;
		return false;
	}

	FExtractor.GetDE(eType, detector);
	FExtractor.GetFD(eType, extractor);

	if (!TestSVM(eType, detector, extractor, trainingSets, testingSets, showImg, verbose))
	{
		cerr << "SVM Test failed!" << endl;
		return false;
	}

	return true;
}

bool FeatherIdentifier::Save(const string &saveName)
{
	if (!trained)
	{
		cerr << "No classifier to save!" << endl;
		return false;
	}

	classifier->save(workingDirectory + saveName);
	return true;
}

bool FeatherIdentifier::Load(const string &loadName)
{
	classifier = SVM::load(workingDirectory + loadName);

	if (classifier == nullptr)
	{
		cerr << "Failed to load classifier! (" << (workingDirectory + loadName) << endl;
		return false;
	}

	trained = true;
	return true;
}

bool FeatherIdentifier::TestSVM(ExtractType eType, Ptr<FeatureDetector> &detector, Ptr<DescriptorExtractor> &extractor, vector<ImageSet> &trainSets, vector<ImageSet> &testSets, bool showImg, bool verbose)
{
	if (!trained)
	{
		cerr << "Cannot Identify, no training data!" << endl;
		return false;
	}

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");		//combine with data from training


	BOWImgDescriptorExtractor BOWide(extractor, matcher);
	BOWide.setVocabulary(vocabulary);
	
	for (int set = 0; set < testSets.size(); set++)
	{
		cout << "Category: " << testSets[set].name << endl;

		int successful = 0;
		for (int im = 0; im < testSets[set].images.size(); im++)
		{
			vector<KeyPoint> keypoints;
			detector->detect(testSets[set].images[im], keypoints);
			KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINTS);

			if (keypoints.empty())
			{
				cout << "No keypoints!" << endl;
				continue;
			}

			Mat descriptor;
			BOWide.compute(testSets[set].images[im], keypoints, descriptor);

			if (descriptor.empty())
			{
				cout << "No descriptor!" << endl;
				continue;
			}

			Mat results;
			float res = classifier->predict(descriptor);

			string predicted = trainSets[(int)res].name;
			string truth = testSets[set].name;

			if (predicted == truth)
				successful++;

			if (verbose)
			{
				if (predicted == truth)
				{
					//cout << " Match!" << endl;
				}
				else
				{
					//cout << "prediction: " << predicted << " (" << res << ") ";
					//cout << "Truth: " << truth << " (" << res << ")";
					//cout << endl;
				}
			}


			if (showImg)
			{
				imshow(predicted, testSets[set].images[im]);
				waitKey(0);
				destroyWindow(predicted);
			}
		}
		cout << "results: " << successful << "/" << testSets[set].images.size() << " successful matches" << endl;
	}

	return true;
}

//Read in a file <dbFile> that describes the name of each set,
//POTENTIAL FORMAT:
//<Extraction method>	<# Words>
//<set name>	<set size>	<directory>	
//<set name>	<set size>	<directory>
bool FeatherIdentifier::MakeTrainingSets(const string &trainingFile, ExtractType &eType, int &numWords)
{
	cout << "Opening training File \"" << workingDirectory + trainingFile << "\"" << endl;
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
		if (!ExtractTypeFromString(methodStr, eType))
		{
			cerr << "Invalid Extract type!" << endl;
			return false;
		}

		if (numWords < MIN_FEATHER_WORDS)
		{
			cerr << "Invalid number of words!" << endl;
			TF.close();
			return false;
		}

		cout << "Training Set: " << methodStr << ", " << numWords << " words" << endl;
	}
	else
	{
		TF.close();
		return false;
	}

	

	int label = 0;
	while (getline(TF, line))
	{
		if (line[0] == '#')			//comment
		{
			continue;
		}

		ImageSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;

		if(!BuildImageSet(subdir, name, label, qty, TS))
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

bool FeatherIdentifier::MakeTestingSets(const string &testingFile)
{
	//try to open the testingFile
	ifstream TF(workingDirectory + testingFile);

	if (!TF.is_open())
	{
		cerr << "Failed to open testingFile! (" << workingDirectory + testingFile << ")" << endl;
		return false;
	}

	//read the first line, should just have the extraction method
	//<SIFT>
	string line;
	stringstream ss;

	int label = 0;
	while (getline(TF, line))
	{
		if (line[0] == '#')			//comment
			continue;

		ImageSet TS;
		string name;
		int qty;
		string subdir;

		ss.clear();
		ss << line;
		ss >> name >> qty >> subdir;

		if(!BuildImageSet(subdir, name, label, qty, TS))
		{
			cerr << "Failed to build testing set for \"" << name << "\"" << endl;
			TF.close();
			return false;
		}

		testingSets.push_back(TS);
		label++;
	}

	TF.close();
	return true;
}

bool FeatherIdentifier::BuildImageSet(const string &subdir, const string &name, const int &label, const int &qty, ImageSet &set)
{
	cout << "Building Set: (" << name << " " << qty << " " << subdir << ")" << endl;
	set.name = name;
	set.label = label;

	for (int i = 0; i < qty; i++)
	{
		string file(workingDirectory + subdir + name + "_" + to_string(i) + ".jpg");
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


bool FeatherIdentifier::CreateVocabulary(Ptr<FeatureDetector> &detector, Ptr<DescriptorExtractor> &extractor)			//further additions: <int numwordss>
{
	if (trainingSets.empty())
	{
		cerr << "training sets empty!" << endl;
		return false;
	}

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
			KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINTS);
			
			if (!keypoints.empty())
			{

				//save a record of the keypoint image
				if (im == 0)
				{
					//cout << "making keypoint image" << endl;
					//make a keypoint image
					Mat keypointImg;
					drawKeypoints(trainingSets[set].images[im], keypoints, keypointImg, Scalar(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
					imwrite(trainingSets[set].name + "_" + StringFromExtractType(eType) + "keypoints.bmp", keypointImg);
				}

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
	BOWKMeansTrainer trainer(numWords);
	trainer.add(trainingDescriptors);
	vocabulary = trainer.cluster();

	//save an image of the vocabulary
	imwrite(StringFromExtractType(eType) + to_string(numWords) + "_vocab.bmp", vocabulary);

	cout << "Done!" << endl;
	return true;
}


bool FeatherIdentifier::CalculateHistograms(Ptr<FeatureDetector> &detector, Ptr<DescriptorExtractor> &extractor, Mat &outSamples, Mat &outLabels)
{
	if (trainingSets.empty())
	{
		cerr << "training sets empty!" << endl;
		return false;
	}

	if (vocabulary.empty())
	{
		cerr << "Vocabulary is empty!" << endl;
		return false;
	}

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
			KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINTS);
			

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

		//cout << "Added " << trainingSets[set].images.size() << " " << trainingSets[set].name << " samples" << endl;
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