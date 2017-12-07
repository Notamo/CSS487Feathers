#include "FeatherIdentifier.h"



FeatherIdentifier::FeatherIdentifier(const string &workingDirectory) : BOWDataMgmt(workingDirectory)
{
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
	cout << "-----------------------------------------" << endl;
	if (!MakeTrainingSets(trainingFile, trainingSets, eType, numWords))
	{
		cerr << "Failed to make training sets!" << endl;
		return false;
	}
	FExtractor.GetDE(eType, detector);
	FExtractor.GetFD(eType, extractor);

	cout << endl << "Creating Vocabulary" << endl;
	cout << "-----------------------------------------" << endl;
	if (!CreateVocabulary(detector, extractor))
	{
		cerr << "Failed to create vocabulary!" << endl;
		return false;
	}

	cout << endl << "Calculating Histograms" << endl;
	cout << "-----------------------------------------" << endl;
	if (!CalculateHistograms(detector, extractor, histograms, labels))
	{
		cerr << "Failed to create histograms!" << endl;
		return false;
	}

	cout << endl << "Training the SVM" << endl;
	cout << "-----------------------------------------" << endl;
	if (!TrainSVM(histograms, labels))
	{
		cerr << "Failed to train SVM!" << endl;
		return false;
	}
	trained = true;

	if (verify)			//test the training set against itself
	{
		cout << endl << "Testing SVM with training data" << endl;
		cout << "-----------------------------------------" << endl;
		TestSVM(eType, detector, extractor, trainingSets, trainingSets, verbose);		
	}

	return true;
}

bool FeatherIdentifier::Identify(const string &testingFile, bool verbose)
{
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;

	cout << endl << "Identifying!" << endl; 
	cout << "-----------------------------------------" << endl;

	if (!trained)
	{
		cerr << "Cannot Identify, no training data!" << endl;
		return false;
	}

	cout << endl << "Making Testing Sets" << endl;
	cout << "-----------------------------------------" << endl;
	if (!MakeTestingSets(testingFile, testingSets))
	{
		cerr << "Failed to make testing set!" << endl;
		return false;
	}

	FExtractor.GetDE(eType, detector);
	FExtractor.GetFD(eType, extractor);

	cout << endl << "Testing Data" << endl;
	cout << "-----------------------------------------" << endl;
	if (!TestSVM(eType, detector, extractor, trainingSets, testingSets, verbose))
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
	
	return SaveSVM(saveName, classifier);
}

bool FeatherIdentifier::Load(const string &loadName)
{
	if (LoadSVM(loadName, classifier))
	{
		trained = true;
		return true;
	}
	else
	{
		trained = false;
		return false;
	}
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
		cout << "Getting " << trainingSets[set].name << " features...";

		for (int im = 0; im < trainingSets[set].images.size(); im++)
		{
			//first get the keypoints
			vector<KeyPoint> keypoints;
			detector->detect(trainingSets[set].images[im], keypoints);
			KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINTS);
			
			if (keypoints.empty())
				continue;

			//Find the descriptors
			Mat descriptors;
			extractor->compute(trainingSets[set].images[im], keypoints, descriptors);

			if (descriptors.empty())
				continue;

			//add he descriptor to the set of trainingDescriptors
			trainingDescriptors.push_back(descriptors);			
		}

		cout << "Done!" << endl;
	}

	if (trainingDescriptors.empty())
	{
		cerr << "No training descriptors were found!" << endl;
		return false;
	}

	cout << "Clustering...";

	//do the actual training to get a vocab
	BOWKMeansTrainer trainer(numWords);
	trainer.add(trainingDescriptors);
	vocabulary = trainer.cluster();

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
		cout << "Making " << trainingSets[set].name << " histograms...";

		for (int im = 0; im < trainingSets[set].images.size(); im++)
		{
			//Find the keypoitns
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
		cout << "Done!" << endl;
	}

	if (samples.empty() || outLabels.empty())
	{
		cerr << "samples are empty!" << endl;
		return false;
	}

	//make sure samples are 32 bit floats for the SVM
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

	cout << "Training...";
	//if we haven't made the classifier yet, do so
	if(classifier.empty())
		classifier = SVM::create();

	Ptr<TrainData> trainData = TrainData::create(samples, ROW_SAMPLE, labels);
	bool result = classifier->trainAuto(trainData);

	cout << "Done!" << endl;
	return result;
}

bool FeatherIdentifier::TestSVM(ExtractType eType, Ptr<FeatureDetector> &detector, Ptr<DescriptorExtractor> &extractor, vector<ImageSet> &trainSets, vector<ImageSet> &testSets, bool verbose)
{
	if (!trained)
	{
		cerr << "Cannot Identify, no training data!" << endl;
		return false;
	}

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor BOWide(extractor, matcher);
	BOWide.setVocabulary(vocabulary);

	for (int set = 0; set < testSets.size(); set++)
	{
		cout << "Category: " << testSets[set].name << endl;

		int successful = 0;
		for (int im = 0; im < testSets[set].images.size(); im++)
		{

			//Get the image's keypoints
			vector<KeyPoint> keypoints;
			detector->detect(testSets[set].images[im], keypoints);
			KeyPointsFilter::retainBest(keypoints, MAX_KEYPOINTS);

			if (keypoints.empty())
				continue;

			//Get the Image Descriptor
			Mat descriptor;
			BOWide.compute(testSets[set].images[im], keypoints, descriptor);

			if (descriptor.empty())
				continue;

			//Run the descriptor through the classifier
			float result = classifier->predict(descriptor);

			//Did we get the right answer?
			string predicted = trainSets[(int)result].name;
			string truth = testSets[set].name;

			if (predicted == truth)
				successful++;


			//Matching info if we want it
			if (verbose)
			{
				cout << "prediction: " << predicted << " (" << result << ") ---";
				cout << "Truth: " << truth << " (" << set << ")" << endl;

				if (predicted == truth)
					cout << "Match!" << endl;
				else
					cout << "Mismatch." << endl;
			}

		}
		cout << "results: " << successful << "/" << testSets[set].images.size() << " successful matches" << endl;
	}

	return true;
}