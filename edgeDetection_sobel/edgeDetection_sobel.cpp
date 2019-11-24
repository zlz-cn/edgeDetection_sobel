#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <cmath>
#include <assert.h>
#include <windows.h>
#include <Commdlg.h>
#include <stdio.h>
#include<exception>

using namespace cv;
using namespace std;

OPENFILENAME ofn;
char szFile[300];

char* ConvertLPWSTRToLPSTR(LPWSTR lpwszStrIn)
{
	LPSTR pszOut = NULL;
	try
	{
		if (lpwszStrIn != NULL)
		{
			int nInputStrLen = wcslen(lpwszStrIn);
			int nOutputStrLen = WideCharToMultiByte(CP_ACP, 0, lpwszStrIn, nInputStrLen, NULL, 0, 0, 0) + 2;
			pszOut = new char[nOutputStrLen];

			if (pszOut)
			{
				memset(pszOut, 0x00, nOutputStrLen);
				WideCharToMultiByte(CP_ACP, 0, lpwszStrIn, nInputStrLen, pszOut, nOutputStrLen, 0, 0);
			}
		}
	}
	catch (std::exception e)
	{
	}

	return pszOut;
}


char* flashreplace(char* file) {
	int size = strlen(file);
	for (int i = 0; i <= size; i++) {
		if (*(file + i) == '\\')
		{
			*(file + i) = '/';
		}
	}
	for (int i = 0; i <= size; i++) {
		printf("%c", *(file + i));
	}


	return file;

}


char* getfile() {
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFile = (LPWSTR)szFile;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = L"All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
	if (GetOpenFileName(&ofn))
	{
		char* getf = ConvertLPWSTRToLPSTR(ofn.lpstrFile);
		getf = flashreplace(getf);
		return getf;
	}
	else
	{
		printf("user cancelled\n");
	}
}

bool sobelEdge(Mat& srcImage, Mat& resultImageX, Mat& resultImageY, uchar threshold)
{
	CV_Assert(srcImage.channels() == 1);
	Mat sobelx = (Mat_<double>(3, 3) << -1, 0,
		1, -2, 0, 2, -1, 0, 1);
	Mat sobely = (Mat_<double>(3, 3) << -1, -2, -1,
		0, 0, 0, 1, 2, 1);
	resultImageX = Mat::zeros(srcImage.rows - 2,
		srcImage.cols - 2, srcImage.type());
	resultImageY = Mat::zeros(srcImage.rows - 2,
		srcImage.cols - 2, srcImage.type());
	double edgeX = 0;
	double edgeY = 0;
	double graMagX = 0;
	double graMagY = 0;
	for (int k = 1; k < srcImage.rows - 1; ++k)
	{
		for (int n = 1; n < srcImage.cols - 1; ++n)
		{
			edgeX = 0;
			edgeY = 0;
			for (int i = -1; i <= 1; ++i)
			{
				for (int j = -1; j <= 1; ++j)
				{
					edgeX += srcImage.at<uchar>(k + i, n + j) *
						sobelx.at<double>(1 + i, 1 + j);
					edgeY += srcImage.at<uchar>(k + i, n + j) *
						sobely.at<double>(1 + i, 1 + j);
				}
			}
			graMagX = sqrt(pow(edgeX, 2));
			graMagY = sqrt(pow(edgeY, 2));
			resultImageX.at<uchar>(k - 1, n - 1) =
				((graMagX > threshold) ? 255 : 0);
			resultImageY.at<uchar>(k - 1, n - 1) =
				((graMagY > threshold) ? 255 : 0);
		}
	}
	return true;
}

int OTSU(Mat& srcImage)
{
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	int threshold = 0;
	double max = 0.0;
	double AvePix[256];
	int nSumPix[256];
	double nProDis[256];
	double nSumProDis[256];
	for (int i = 0; i < 256; i++)
	{
		AvePix[i] = 0.0;
		nSumPix[i] = 0;
		nProDis[i] = 0.0;
		nSumProDis[i] = 0.0;
	}
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (double)nSumPix[i] / (nRows * nCols);

	}
	AvePix[0] = 0;
	nSumProDis[0] = nProDis[0];
	for (int i = 1; i < 256; i++)
	{
		nSumProDis[i] = nSumProDis[i - 1] + nProDis[i];
		AvePix[i] = AvePix[i - 1] + i * nProDis[i];
	}
	double mean = AvePix[255];
	for (int k = 1; k < 256; k++)
	{
		double PA = nSumProDis[k];
		double PB = 1 - nSumProDis[k];
		double value = 0.0;
		if (fabs(PA) > 0.001 && fabs(PB) > 0.001)
		{
			double MA = AvePix[k];
			double MB = (mean - PA * MA) / PB;
			value = value = (double)(PA * PB * pow((MA - MB), 2));			
			if (value > max)
			{
				max = value;
				threshold = k;
			}
		}
	}
	return threshold;
}





int main()
{
	char* opfile = getfile();
	Mat srcImage = cv::imread(opfile);
	if (!srcImage.data)
		return -1;
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	namedWindow("srcGray", CV_WINDOW_NORMAL);
	imshow("srcGray", srcGray);
	int otsuThreshold = OTSU(srcGray);
	cout << otsuThreshold << endl;
	Mat XresultImage;
	Mat YresultImage;
	sobelEdge(srcGray, XresultImage, YresultImage, otsuThreshold);
	Mat resultImage;
	addWeighted(XresultImage, 0.5, YresultImage, 0.5, 0.0, resultImage);
	std::vector<Vec2f>lines;
	HoughLines(resultImage, lines, 1, CV_PI / 180, 90);
	std::vector<cv::Vec2f>::const_iterator begin = lines.begin();
	while (begin != lines.end())
	{
		float rho = (*begin)[0];
		float theta = (*begin)[1];
		printf("%.1f=sin%.1f+cos%.1f\n", rho, theta, theta);
		if (theta < CV_PI / 4. || theta>3. * CV_PI / 4.)
		{
			cv::Point pt1(0, (rho / sin(theta)));
			cv::Point pt2(resultImage.rows, (rho - resultImage.rows * cos(theta)) / sin(theta));
			cv::line(srcImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);
		}
		else
		{
			cv::Point pt1(rho / cos(theta), 0);
			cv::Point pt2((rho - resultImage.cols * sin(theta)) / cos(theta), resultImage.cols);
			cv::line(srcImage, pt1, pt2, cv::Scalar(255, 0, 255), 1);
		}
		begin++;
	}



	namedWindow("resx", CV_WINDOW_NORMAL);
	namedWindow("resy", CV_WINDOW_NORMAL);
	namedWindow("res", CV_WINDOW_NORMAL);
	namedWindow("src", CV_WINDOW_NORMAL);
	imshow("resx", XresultImage);
	imshow("resy", YresultImage);
	imshow("res", resultImage);
	imshow("src", srcImage);

	waitKey(0);
	return 0;
}