﻿// edgeDetection_sobel.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <iostream>  
#include <cmath>
#include <assert.h>
using namespace cv;
using namespace std;
bool sobelEdge(Mat& srcImage, Mat& resultImageX, Mat& resultImageY, uchar threshold)
{
	CV_Assert(srcImage.channels() == 1);
	// 初始化水平核因子
	Mat sobelx = (Mat_<double>(3, 3) << -1, 0,
		1, -2, 0, 2, -1, 0, 1);
	// 初始化垂直核因子
	Mat sobely = (Mat_<double>(3, 3) << -1, -2, -1,
		0, 0, 0, 1, 2, 1);
	resultImageX = Mat::zeros(srcImage.rows - 2,
		srcImage.cols - 2, srcImage.type());
	resultImageY = Mat::zeros(srcImage.rows - 2,
		srcImage.cols - 2, srcImage.type());
	double edgeX = 0;
	double edgeY = 0;
	double graMagX = 0;// 垂直方向上的梯度模长
	double graMagY = 0;// 水平方向上的梯度模长
	for (int k = 1; k < srcImage.rows - 1; ++k)
	{
		for (int n = 1; n < srcImage.cols - 1; ++n)
		{
			edgeX = 0;
			edgeY = 0;
			// 遍历计算水平与垂直梯度
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
			// 计算垂直方向上的梯度模长
			graMagX = sqrt(pow(edgeX, 2));

			// 计算水平方向上的梯度模长
			graMagY = sqrt(pow(edgeY, 2));
			// 二值化
			resultImageX.at<uchar>(k - 1, n - 1) =
				((graMagX > threshold) ? 255 : 0);

			// 二值化
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
			double MA = AvePix[k];//前一半的平均
			double MB = (mean - PA * MA) / PB;//后一半的平均
			value = value = (double)(PA * PB * pow((MA - MB), 2));//类间方差  
				 //或者这样value = (double)(PA * PB * pow((MA-MB),2));//类间方差
			 //pow(PA,1)* pow((MA - mean),2) + pow(PB,1)* pow((MB - mean),2)
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
	Mat srcImage = cv::imread("building.jpg");
	if (!srcImage.data)
		return -1;
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	imshow("srcGray", srcGray);
	//调用二值化函数得到最佳阈值
	int otsuThreshold = OTSU(srcGray);
	cout << otsuThreshold << endl;
	Mat XresultImage;
	Mat YresultImage;
	sobelEdge(srcGray, XresultImage, YresultImage, otsuThreshold);
	Mat resultImage;
	//水平垂直边缘叠加
	addWeighted(XresultImage, 0.5, YresultImage, 0.5, 0.0, resultImage);
	imshow("resx", XresultImage);
	imshow("resy", YresultImage);
	imshow("res", resultImage);
	waitKey(0);
	return 0;
}



// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
