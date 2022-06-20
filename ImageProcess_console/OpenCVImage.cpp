#include "OpenCVImage.h"
using namespace std;
using namespace cv;



OpenCVImage::OpenCVImage(std::vector<std::string> filesPath, std::string savePath, std::string prefix)
{
	m_filesPath = filesPath;
	m_savePath = savePath;
	m_prefix = prefix;
}

void OpenCVImage::run()
{
	for (int x = 0; x < m_filesPath.size(); x++) {
		cv::Mat image;
		image = cv::imread(m_filesPath[x]);

		if(image.rows > 0 && image.cols >0)
			ImageProcessing(image, x,m_prefix);
	}
}

void OpenCVImage::ImageProcessing(cv::Mat src,int count, string prefix)
{
	cv::Mat dst,buf,result,roi,th,th2, gaussianBlur;
	cv::resize(src, src, cv::Size(), 0.5,0.5);

	src.copyTo(dst);
	GaussianBlur(dst, dst, cv::Size(3, 3), 0);
	dst.copyTo(gaussianBlur);

	/*Rect roi_rect;

	roi_rect = getObjRoi(dst, 100);
	dst(roi_rect).copyTo(result);

	saveImage(result, count, "stain");*/

	cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
	cv::threshold(dst, th, 55, 255, 0);

	//saveImage(th, count, "th");


	gaussianBlur(getObjRoi(th, 100)).copyTo(roi);
	cv::cvtColor(roi, th2, cv::COLOR_BGR2GRAY);
	cv::threshold(th2, th2, 55, 255, 0);
	//saveImage(th2, count, "th2");

	Mat edge_th = getEdgeImage(getEdgeMag(roi));
	
	Mat edge_data = getEdgeMag(roi);
	
	Mat edge_nom;
	normalize(edge_data, edge_nom, 0, 255 , NORM_MINMAX);

	//saveImage(edge_th, count, "edge");
	//saveImage(edge_nom, count, "edge_nom");
	saveImage(roi, count, prefix + "_DOT");


	//GaussianBlur(roi, dst, cv::Size(3, 3), 0);

	//saveImage(roi, count, prefix + "_OK");

	//result = resoultionInspection(roi,count);
	//saveImage(result,count,m_prefix + "edge_th");

	contours(th2, count);
	
}

void OpenCVImage::saveImage(cv::Mat src, int count, std::string modelName)
{
	std::string fileName = m_savePath + "\\" + std::to_string(count) + "_" + modelName + ".bmp";

	cv::Mat saveImage;
	if (src.channels() != 3)
	{
		std::vector<cv::Mat> buf(3);
		buf[0] = src;
		buf[1] = src;
		buf[2] = src;
		merge(buf, saveImage);
	}
	else
		saveImage = src;

	cv::imwrite(fileName, saveImage);

}

void OpenCVImage::contours(cv::Mat src, int count)
{
	cv::Mat buf, convert;

	src.copyTo(convert);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//convert.convertTo(convert, CV_32SC1);

	//find_contours(0, 0, src);

	//cv::findContours(convert, contours, hierarchy, cv::RETR_FLOODFILL, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(convert, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	cv::Mat drawing = cv::Mat(src.size(), CV_8UC1, cv::Scalar::all(0));
	
	int flag = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		
		if (area > 30) {
			//putText(drawing, to_string(flag), cv::Point(contours[i][0].x, contours[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);
			flag++;
			//putText(drawing, to_string(area), cv::Point(contours[i][0].x, contours[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(0), 1);
			//cv::drawContours(drawing, contours, i, cv::Scalar::all(255), 1,FILLED, hierarchy);
			cv::drawContours(drawing, contours, i, cv::Scalar::all(255), 1);

		}
	}

	//buf = src(getObjRoi(drawing, 100));


	//saveImage(drawing, count, "drawing");
	//saveImage(buf, count, "ROI");

}

cv::Rect OpenCVImage::getObjRoi(cv::Mat src, int margin)
{
	Rect roi_object;
	int startX, startY, endX, endY;
	uchar* data = (uchar*)src.data;
	startX = src.cols;
	startY = src.rows;
	endX = 0;
	endY = 0;

	for (int idy = 0; idy < src.rows; idy++)
	{
		for (int idx = 0; idx < src.cols; idx++)
		{
			if (data[idy * src.cols + idx] >= 1)
			{
				if (startX > idx)
					startX = idx;
				if (startY > idy)
					startY = idy;
				if (endX < idx)
					endX = idx;
				if (endY < idy)
					endY = idy;
			}
		}
	}


	startX -= margin;
	startY -= margin;
	endX += margin;
	endY += margin;

	if (startX < 0)
		startX = 0;
	if (startY < 0)
		startY = 0;
	if (endX >= src.cols)
		endX = src.cols - 1;
	if (endY >= src.rows)
		endY = src.rows - 1;

	roi_object.x = startX;
	roi_object.y = startY;
	roi_object.width = endX - startX; //cols
	roi_object.height = endY - startY; //row


	return roi_object;
}

void OpenCVImage::contoursObjectDetection(cv::Mat src, int count)
{
	cv::Mat buf;

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	cv::findContours(src, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);

		//if (m_find_contours[i].size() > 1000 && m_find_contours[i].size() < 3000 && flag < 2) 
		if (area > 3000)
		{
			Rect temp_rect = cv::boundingRect(contours[i]);

			temp_rect.x -= 100;
			temp_rect.y -= 100;
			temp_rect.width += 200;
			temp_rect.height += 200;

			buf = src(temp_rect);
		}
	}

	saveImage(buf, count, "VW310_TEST_OK");
}

void OpenCVImage::draw_and_fill_contours(std::vector<std::vector<cv::Point>>& contours, std::vector<std::vector<cv::Point>>& hull, std::vector<cv::Vec4i>& hierarchy, cv::Mat img_gray)
{
	cv::Mat contours_result = cv::Mat::zeros(img_gray.size(), CV_8UC1);
	cv::Mat fill_contours_result = cv::Mat::zeros(img_gray.size(), CV_8UC1);

	for (unsigned int i = 0, n = contours.size(); i < n; ++i)
	{
		cv::Scalar color = cv::Scalar(255, 255, 255);
		cv::drawContours(contours_result, contours, i, color, 4, 8, hierarchy, 0, cv::Point());
	}

	cv::fillPoly(fill_contours_result, hull, cv::Scalar(255, 255, 255));

	saveImage(contours_result, 0, "result");
	saveImage(fill_contours_result, 0, "fill");

	//cv::imshow("Contours Result", contours_result);
	//cv::imshow("Fill Contours Result", fill_contours_result);
}

void OpenCVImage::find_contours(int, void*, cv::Mat img_gray)
{
	cv::Mat canny_output;
	//cv::Canny(img_gray, canny_output, thresh, thresh * 2);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img_gray, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point>> hull(contours.size());
	for (unsigned int i = 0, n = contours.size(); i < n; ++i) {
		cv::convexHull(cv::Mat(contours[i]), hull[i], false);
	}

	draw_and_fill_contours(contours, hull, hierarchy, img_gray);
}

cv::Mat OpenCVImage::resoultionInspection(cv::Mat src , int count)
{
	cv::Mat edge_bin,edge_Image, nomarlize_edge;
	float edge_data = 0;

	edge_bin = getEdgeMag(src);

	//영역을 어떻게 나눌지 
	edge_data = getEdgeData(edge_bin);

	threshold(edge_bin, edge_bin, 0.1, 255, cv::THRESH_TOZERO);

	normalize(edge_bin, nomarlize_edge, 0, 255, NORM_MINMAX);

	saveImage(nomarlize_edge, count, m_prefix + "edge_nom");

	edge_Image = getEdgeImage(edge_bin);

	return edge_Image;
}

float OpenCVImage::getEdgeData(cv::Mat src)
{
	float* edgeData = (float*)src.data;
	float srcData;

	float sum = 0;
	float mean = 0;
	int count = 0;
	float count2 = 0;
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			srcData = edgeData[y * src.cols + x];

			if (srcData > 0.1 && srcData < 0.4)
			{
				sum += srcData;
				count++;
			}

			if (srcData > 0.1)
				count2++;
		}

	}

	if (count > 0)
		mean = count / count2;

	return mean;
}

cv::Mat OpenCVImage::getEdgeImage(cv::Mat src)
{
	cv::Mat dst,edge;
	src.copyTo(dst);

	threshold(dst, edge, 0.1, 255, 0);

	return edge;
}

cv::Mat OpenCVImage::getEdgeMag(cv::Mat src)
{
	Mat g_src, f_src;
	Mat edge_hor, edge_ver, edge_diag, edge_adiag;
	Mat edge_mag;

	if (src.channels() != 1)
		cvtColor(src, g_src, COLOR_BGR2GRAY);
	else
		src.copyTo(g_src);
	if (src.depth() != CV_32FC1)
		g_src.convertTo(f_src, CV_32FC1, 1.0 / 255.0);
	else
		g_src.copyTo(f_src);

	GaussianBlur(f_src, f_src, cv::Size(3, 3), 0);


	edge_hor = matSobelOperation(f_src, SOBEL_HOR);
	edge_ver = matSobelOperation(f_src, SOBEL_VER);
	edge_diag = matSobelOperation(f_src, SOBEL_DIAG);
	edge_adiag = matSobelOperation(f_src, SOBEL_ADIAG);

	pow(edge_hor, 2, edge_hor);
	pow(edge_ver, 2, edge_ver);
	pow(edge_diag, 2, edge_diag);
	pow(edge_adiag, 2, edge_adiag);
	edge_mag = edge_hor + edge_ver + edge_diag + edge_adiag;
	sqrt(edge_mag, edge_mag);

	/*imwrite("edge_hor.jpg", edge_hor);
	imwrite("edge_ver.jpg", edge_ver);
	imwrite("edge_diag.jpg", edge_diag);
	imwrite("edge_adiag.jpg", edge_adiag);*/

	return edge_mag;
}

cv::Mat OpenCVImage::matSobelOperation(Mat src, E_SOBEL_DIR dir)
{
	Mat kernel(3, 3, CV_32F, Scalar(0));
	Mat dst;

	switch (dir)
	{
	case SOBEL_HOR:
		kernel.at<float>(0, 0) = -1;
		kernel.at<float>(1, 0) = -1;
		kernel.at<float>(2, 0) = -1;
		kernel.at<float>(0, 2) = 1;
		kernel.at<float>(1, 2) = 1;
		kernel.at<float>(2, 2) = 1;
		break;
	case SOBEL_VER:
		kernel.at<float>(0, 0) = -1;
		kernel.at<float>(0, 1) = -1;
		kernel.at<float>(0, 2) = -1;
		kernel.at<float>(2, 0) = 1;
		kernel.at<float>(2, 1) = 1;
		kernel.at<float>(2, 2) = 1;
		break;
	case SOBEL_DIAG:
		kernel.at<float>(0, 1) = 1;
		kernel.at<float>(0, 2) = 2;
		kernel.at<float>(1, 2) = 1;
		kernel.at<float>(1, 0) = -1;
		kernel.at<float>(2, 0) = -2;
		kernel.at<float>(2, 1) = -1;
		break;
	case SOBEL_ADIAG:
		kernel.at<float>(1, 2) = 1;
		kernel.at<float>(2, 2) = 2;
		kernel.at<float>(2, 1) = 1;
		kernel.at<float>(0, 1) = -1;
		kernel.at<float>(0, 0) = -2;
		kernel.at<float>(1, 0) = -1;
		break;
	default:
		kernel.at<float>(0, 0) = -1;
		kernel.at<float>(1, 0) = -1;
		kernel.at<float>(2, 0) = -1;
		kernel.at<float>(0, 2) = 1;
		kernel.at<float>(1, 2) = 1;
		kernel.at<float>(2, 2) = 1;
		break;
	}
	filter2D(src, dst, src.depth(), kernel);
	return dst;
}
