#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
using namespace cv;

#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
	Mat img1 = imread("test.jpg");

	if (!img1.data)
	{
		cout << "Could not open test.jpg!" << endl;
		return -1;
	}

	namedWindow("window1");
	imshow("window1", img1);

	waitKey(0);
	return 0;
}