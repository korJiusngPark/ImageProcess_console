#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <vector>
#include <string>


enum E_SOBEL_DIR
{
	SOBEL_HOR = 0,
	SOBEL_VER,
	SOBEL_DIAG,
	SOBEL_ADIAG,
};

class OpenCVImage
{


public:
	OpenCVImage(std::vector<std::string> filesPath, std::string savePath, std::string prefix);

public:
	void run();
	void ImageProcessing(cv::Mat src,int count, std::string prefix);
	void saveImage(cv::Mat src, int count, std::string modelName);

	void contours(cv::Mat src, int count);
	cv::Rect getObjRoi(cv::Mat src, int margin);
	void contoursObjectDetection(cv::Mat src, int count);

	void draw_and_fill_contours(std::vector<std::vector<cv::Point>>& contours,
		std::vector<std::vector<cv::Point>>& hull,
		std::vector<cv::Vec4i>& hierarchy, cv::Mat img_gray, int count);

	void find_contours( cv::Mat img_gray,int count);

	cv::Mat resoultionInspection(cv::Mat src, int count);
	float getEdgeData(cv::Mat src);
	cv::Mat getEdgeImage(cv::Mat src);
	cv::Mat getEdgeMag(cv::Mat src);

	cv::Mat matSobelOperation(cv::Mat src, E_SOBEL_DIR dir);
	float getROiEdgeData(cv::Mat edge_data);
	int getROiEdgeData(cv::Mat edge_data,int x);

	bool Resolution(cv::Mat src, int cols, int rows, int count);
	cv::Mat contoursCheck(cv::Mat src, int countoursCnt, int sizeMin, int sizeMax,int count);

	void dotInspection(cv::Mat src,int count);
	cv::Mat dotarea(cv::Mat src);

	cv::Mat m_tar_gray_obj, m_edge_data, m_result;
public:
	std::vector<std::string> m_filesPath;
	std::string m_savePath;
	std::string m_prefix;
};

