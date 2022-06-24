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
	//cv::resize(src, src, cv::Size(), 0.5,0.5);

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

	dotInspection(src,count);

	//Mat edge_th = getEdgeImage(getEdgeMag(roi));
	//
	//Mat edge_data = getEdgeMag(roi);
	//
	//Mat edge_nom;
	//normalize(edge_data, edge_nom, 0, 255 , NORM_MINMAX);

	////saveImage(edge_th, count, "edge");
	////saveImage(edge_nom, count, "edge_nom");
	//saveImage(roi, count, prefix + "_DOT2");


	//Resolution(roi,2,2, count);

	////GaussianBlur(roi, dst, cv::Size(3, 3), 0);

	////saveImage(roi, count, prefix + "_OK");

	//result = resoultionInspection(roi,count);
	////saveImage(result,count,m_prefix + "edge_th");
	//contours(th2, count);
	//contoursCheck(roi, 1, 1, 1, count);
}

void OpenCVImage::saveImage(cv::Mat src, int count, std::string modelName)
{
	std::string fileName = m_savePath + "\\" + std::to_string(count) + "_" + modelName + ".jpg";

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

void OpenCVImage::draw_and_fill_contours(std::vector<std::vector<cv::Point>>& contours, std::vector<std::vector<cv::Point>>& hull, std::vector<cv::Vec4i>& hierarchy, cv::Mat img_gray,int count)
{
	cv::Mat contours_result = cv::Mat::zeros(img_gray.size(), CV_8UC1);
	cv::Mat fill_contours_result = cv::Mat::zeros(img_gray.size(), CV_8UC1);

	for (size_t i = 0, n = contours.size(); i < n; ++i)
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

void OpenCVImage::find_contours(cv::Mat img_gray,int count)
{
	cv::Mat canny_output;
	//cv::Canny(img_gray, canny_output, thresh, thresh * 2);
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img_gray, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point>> hull(contours.size());
	for (size_t i = 0, n = contours.size(); i < n; ++i) {
		cv::convexHull(cv::Mat(contours[i]), hull[i], false);
	}

	draw_and_fill_contours(contours, hull, hierarchy, img_gray,count);
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


float OpenCVImage::getROiEdgeData(cv::Mat edge_data)
{
	Mat dst(edge_data.size(), CV_8UC1);

	uchar* dstData = (uchar*)dst.data;
	float* edgeData;
	float srcData;

	float sum = 0;
	float mean = 0;
	int count = 0;
	float count2 = 0;
	for (int y = 0; y < edge_data.rows; y++)
	{
		for (int x = 0; x < edge_data.cols; x++)
		{
			//srcData = edgeData[y * edge_data.cols + x];
			edgeData = edge_data.ptr<float>(y);

			if (edgeData[x] > 0.1 && edgeData[x] < 1)
			{
				dstData[y * dst.cols + x] = 0;
				sum += edgeData[x];
				count++;
			}
			else
				dstData[y * dst.cols + x] = 255;

		}

	}

	if (count > 200)
		mean = sum / count;

	return mean;
}

int OpenCVImage::getROiEdgeData(cv::Mat edge_data, int x)
{
	Mat dd;
	Mat dst(edge_data.size(), CV_8UC1);
	threshold(edge_data, dd, 50, 255, 0);
	uchar* dstData;
	uchar* edgeData;
	int srcData;

	Scalar ff = mean(edge_data);

	float sum = 0;
	float mean = 0;
	int count = 0;
	float count2 = 0;
	for (int y = 0; y < edge_data.rows; y++)
	{
		for (int x = 0; x < edge_data.cols; x++)
		{
			edgeData = edge_data.ptr<uchar>(y);
			dstData = dst.ptr<uchar>(y);

			if (edgeData[x] > 10 && edgeData[x] < 255)
			{
				dstData[x] = 0;
				sum += edgeData[x];
				count++;
			}
			else
				dstData[x] = 255;


			//if (srcData > 0.1)
			//	count2++;
		}

	}

	if (count > 15)
		mean = sum / count;

	//imshow("dd", dst);
	//imshow("22", dd);

	waitKey(0);

	return float(count);
}

bool OpenCVImage::Resolution(cv::Mat src, int cols, int rows,int count)
{
	Mat edge = getEdgeMag(src);
	Mat edge_th;
	threshold(edge, edge_th, 0.08, 255, 0);
	//saveImage(edge_th, _T("edge"), (int)m_param[PARAM_ENABLE_SAVE_IMG_PROC].param, 1, false);

	if (cols == NULL || rows == NULL)
	{
		cols = 5;
		rows = 5;
	}

	bool re = true;

	Mat result, resultInform;
	src.copyTo(result);
	src.copyTo(resultInform);

	int cols_width = src.cols / cols;
	int rows_width = src.rows / rows;



	for (int x = 0; x < cols; x++) {
		for (int y = 0; y < rows; y++)
		{
			int paramCnt = ((y * rows) + x);

			cv::Rect rect(x * cols_width, y * rows_width, cols_width, rows_width);
			float edge_result = getROiEdgeData(edge(rect));
			//float edge_result = getROiEdgeData(m_ImgNor(rect),1);


			cv::rectangle(result, rect, Scalar(255, 255, 255), 1);
			putText(result, to_string(edge_result).substr(0, 4), cv::Point((x * cols_width) + 10, ((y * rows_width) + (y + 1) * rows_width) / 2), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			putText(result, to_string(paramCnt + 1).substr(0, 4), cv::Point((x * cols_width) + 20, (((y * rows_width) + (y + 1) * rows_width) / 2) + 20), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);

		}
	}

	saveImage(result, count,"해상력");


	return re;
}

cv::Mat OpenCVImage::contoursCheck(cv::Mat src, int countoursCnt, int sizeMin, int sizeMax,int count)
{
	vector<vector<cv::Point>> contours;
	Mat gray, test, gr, dst, erode, dilate, dst2;
	//	vector<Point2f> approx;

	//m_tarAreaSize = 0;
	//m_tarAreaSizeSum = 0;

	cvtColor(src, dst, COLOR_BGR2GRAY);
	threshold(dst, gray, 50, 255, 0);
	threshold(dst, test, 50, 255, THRESH_TOZERO);
	cv::Mat element5(3, 3, CV_8U, cv::Scalar(1));

	saveImage(gray, count, "gray");
	//saveImage(test, count, "test");

	//find_contours(gray,count);


	morphologyEx(test, gr, MORPH_GRADIENT, element5);
	threshold(gr, gr, 1, 255, 0);
	//saveImage(gr, count,"gradient");


	morphologyEx(gr, gr, MORPH_CLOSE, element5);
	gray.convertTo(gray, CV_8UC1);
	//saveImage(gr,count, "close");

	gr.copyTo(dst2);


	morphologyEx(dst2, gr, MORPH_OPEN, element5);
	//saveImage(gr, count, "open");

	morphologyEx(dst2, erode, MORPH_ERODE, element5);
	//saveImage(erode, count, "erode");

	morphologyEx(dst2, dilate, MORPH_DILATE, element5);
//	saveImage(dilate, count, "dilate");

	//saveImage(gray, _T("gray "), 1, 1, false);
	//saveImage(test, _T("test "), 1, 1, false);

	cv::findContours(gray, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(src.size(), CV_8UC3);
	Mat drawing1 = Mat::zeros(src.size(), CV_8UC1);

	int flag = 0;

	vector<vector<cv::Point>> approx(contours.size());

	
	int m_find_contours_cnt_flag = 0;

	drawing = Mat::zeros(src.size(), CV_8UC1);

	for (int i = 0; i < contours.size(); i++)
	{
	
		int area = contourArea(contours[i]);

		approxPolyDP(Mat(contours[i]), approx[i], arcLength(Mat(contours[i]), true) * 0.01, true);


		if (area > 50) {
			flag++;
			Rect temp_rect = cv::boundingRect(contours[i]);

			//m_contours_rect.push_back(temp_rect);
			//m_contours.push_back(m_find_contours[i]);


			cv::drawContours(drawing, contours, i, cv::Scalar::all(255), 1);
			/*cv::drawContours(drawing1, contours, i, cv::Scalar::all(255), 1);
			putText(drawing1, std::to_string(flag), Point(contours[i][0].x - 100, contours[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			putText(drawing1, std::to_string(area), Point(contours[i][0].x - 100, contours[i][0].y + 100), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			saveImage(drawing1, num, 1, 1, false);*/


			//putText(drawing, std::to_string(m_find_contours_cnt_flag), Point(m_find_contours[i][0].x - 50, m_find_contours[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			putText(drawing, std::to_string(i), Point(contours[i][0].x - 100, contours[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			putText(drawing, std::to_string(area), Point(contours[i][0].x - 100, contours[i][0].y + 100), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);

		}

		for (int i = 0; i < contours[0].size(); i++)
		{
			//cv::drawContours(drawing1, approx, i, cv::Scalar::all(255), 1);
			for (int x = 0; x < approx[0].size(); x++) {
				putText(drawing, std::to_string(i), Point(approx[0][x].x, approx[0][x].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
			}

		}

		//cv::drawm_find_contours(drawing, approx, i, cv::Scalar::all(255), 1);
	}


	for (int i = 0; i < approx.size(); i++)
	{
		cv::drawContours(drawing1, approx, i, cv::Scalar::all(255), 1);
		for (int x = 0; x < approx[0].size(); x++) {
			putText(drawing1, std::to_string(x), Point(approx[0][x].x, approx[0][x].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
		}

	}
	putText(drawing1, std::to_string(approx.size()), Point(100, 100), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);

	saveImage(drawing, count, "drawing");

	saveImage(drawing1, count, "approx");

	//T("drawing 111"), 1, 1, false);

	//m_tarContoursSize = flag;

	return drawing;
}
int lowTh = 50;
int highTh = 150;
Mat m_canny,m_src;
static void CannyThreshold(int, void*) 
{
	//m_src.copyTo(m_canny);
	blur(m_src, m_canny, Size(3, 3));
	Canny(m_canny, m_canny, lowTh, highTh, 3);
	imshow("canny", m_canny);
}
void OpenCVImage::dotInspection(cv::Mat src, int count)
{
	Mat dst, gray,edge;
	src.copyTo(m_result);
	cvtColor(src, dst, COLOR_BGR2GRAY);
	GaussianBlur(dst, dst, Size(3, 3),0);

	dst.copyTo(m_tar_gray_obj);
	threshold(dst, gray, 60, 255, 0);
	saveImage(src, count, "src");

	saveImage(gray, count, "gray");

	Mat gray_zero,edge_tozero;
	threshold(dst, gray_zero,60, 255, THRESH_TOZERO);

	edge = getEdgeMag(dst);
	edge_tozero = getEdgeMag(gray_zero);

	edge.copyTo(m_edge_data);
	Mat edge_th,edge_th2;
	threshold(edge, edge_th, 0.1, 255, 0);
	threshold(edge_tozero, edge_th2, 0.2, 255, 0);

	saveImage(gray_zero, count, "gray_zero");
	saveImage(edge_th2, count, "edge_th2");


	Mat edge_nom;
	normalize(edge, edge_nom, 0, 255 , NORM_MINMAX);
	saveImage(edge_nom, count, "edge_nom");

	threshold(edge_nom, edge_nom,10, 255, THRESH_TOZERO);
	saveImage(edge_th, count, "edge_th");

	float edge_result = getROiEdgeData(edge);

	vector<vector<cv::Point>> contours;

	cv::findContours(gray, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	vector<vector<cv::Point>> approx(contours.size());


	Mat drawing = Mat::zeros(src.size(), CV_8UC1);	Mat drawing2 = Mat::zeros(src.size(), CV_8UC1);
	Mat drawing3 = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approx[i], arcLength(Mat(contours[i]), true) * 0.001, true);

		int area = contourArea(contours[i]);

		if (area > 50) {
		
			cv::drawContours(drawing, contours, i, cv::Scalar::all(255), 2);
			cv::drawContours(drawing2, contours, i, cv::Scalar::all(255), 2);

			if (edge_result > 0.4)
				cv::drawContours(drawing3, contours, i, cv::Scalar::all(255), 12);
			else
				cv::drawContours(drawing3, contours, i, cv::Scalar::all(255), 16);

		}
	}
	putText(drawing2, to_string(contours.size()), cv::Point(50,50), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

	//	for (int i = 0; i < contours.size(); i++)
	//	{
	//		int flag = 0;

	//		//cv::drawContours(drawing1, approx, i, cv::Scalar::all(255), 1);
	//		for (int x = 0; x < approx[i].size(); x++) {
	//			flag++;
	//			//putText(drawing2, std::to_string(flag), Point(approx[i][x].x, approx[i][x].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
	//		}
	//		putText(drawing2, std::to_string(flag), Point(50, i * 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);

	//	}

	Mat test,tt,tt3;
	Mat canny;
	/*dst = imread("C:\\Users\\USER\\Desktop\\VW316 dot\\검출안되는애들\\122_src.bmp");

	cvtColor(dst, dst, COLOR_BGR2GRAY);
	GaussianBlur(dst, dst, Size(3, 3),0);

	m_src = dst;

	namedWindow("Canny edge", WINDOW_AUTOSIZE);
	createTrackbar("Min th", "Canny edge", &lowTh, 1000, CannyThreshold);
	createTrackbar("max th", "Canny edge", &highTh, 1000, CannyThreshold);
	CannyThreshold(0, 0);
	waitKey(0);*/

	Canny(dst, canny, 7, 20);
	canny.depth();
	saveImage(canny, count, "canny");

	//saveImage(canny, count, "canny");
	Mat convertNom;
	edge_nom.convertTo(convertNom,CV_8UC1);

	edge_th.convertTo(edge_th, CV_8UC1);

	test = canny - drawing;
	tt = convertNom - drawing3;
	tt3 = edge_th - drawing3;
	saveImage(test, count, "test");
	saveImage(drawing, count, "drawing");
	saveImage(drawing2, count, "drawing2");
	saveImage(tt, count, "nom");
	saveImage(tt3, count, "th");

	vector<vector<cv::Point>> contours_edge;

	edge_th2.convertTo(edge_th2, CV_8UC1);

	cv::findContours(gray_zero, contours_edge, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	Mat drawing4 = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < contours_edge.size(); i++)
	{
		int area = contourArea(contours_edge[i]);

		cv::drawContours(drawing4, contours_edge, i, cv::Scalar::all(255), 1);
	}
	putText(drawing4, to_string(contours_edge.size()), cv::Point(50, 50), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

	saveImage(drawing4, count, "drawing4");

	gray.channels();
	edge_th.channels();

	gray.depth();
	edge_th.depth();


	Mat dot = Mat(gray.size(), CV_8UC1);
	Mat dot2 = Mat(gray.size(), CV_8UC1);

	uchar* dstData;
	uchar* grayData;
	uchar* resultData;
	float* edgeNomData;
	uchar* edgeThData;

	uchar* dot2Data;
	int srcData;

	float sum = 0;
	float mean = 0;
	float count2 = 0;
	for (int y = 0; y < gray.rows; y++)
	{
		for (int x = 0; x < gray.cols; x++)
		{
			grayData = gray.ptr<uchar>(y);
			dstData = test.ptr<uchar>(y);
			resultData = dot.ptr<uchar>(y);
			edgeNomData = edge_nom.ptr<float>(y);
			dot2Data = dot2.ptr<uchar>(y);
			edgeThData = tt3.ptr<uchar>(y);
			if (grayData[x] == 255 && dstData[x] == 255)
			{
				resultData[x] = 255;

			}
			else
			{
				resultData[x] = 0;
			}

			if (grayData[x] == 255 && edgeThData[x] == 255)
			{
				dot2Data[x] = 255;
			}
			else
			{
				dot2Data[x] = 0;
			}
		}

	}

	//saveImage(gray, count, "gray");
	//saveImage(edge_th, count, "edge");
	saveImage(dot, count, "result");
	cv::Mat element5(3, 3, CV_8U, cv::Scalar(1));

	morphologyEx(dot2, dot2, MORPH_DILATE,element5);
	saveImage(dot2, count, "dot2");

	
	vector<vector<cv::Point>> contours2;
	vector<vector<cv::Point>> contours3;

	cv::findContours(dot, contours2, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(dot2, contours2, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);


	Mat result_drawing = Mat::zeros(src.size(), CV_8UC1);
	Mat result_drawing2 = Mat::zeros(src.size(), CV_8UC1);
	for (int i = 0; i < contours2.size(); i++)
	{

		int area = contourArea(contours2[i]);

		if (area > 2) {

			cv::drawContours(result_drawing, contours2, i, cv::Scalar::all(255), 2);
			putText(result_drawing, to_string(area), cv::Point(contours2[i][0].x, contours2[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

		}
	}

	for (int i = 0; i < contours3.size(); i++)
	{

		int area = contourArea(contours3[i]);

		if (area > 1) {

			cv::drawContours(result_drawing2, contours3, i, cv::Scalar::all(255), 2);
			putText(result_drawing2, to_string(area), cv::Point(contours3[i][0].x, contours3[i][0].y), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

		}
	}

	saveImage(result_drawing, count, "canny_result");

	Mat result1, reulst2;
	result1 = dotarea(dot);
	putText(result1, to_string(edge_result), cv::Point(50,50), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

	saveImage(result1, count, "canny2");

	reulst2 = dotarea(dot2);
	saveImage(reulst2, count, "reulst2");

	//saveImage(result_drawing2, count, "sobel_result");

	//putText(drawing2, to_string(contours.size()), cv::Point(50,50), FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar::all(255), 1);

	//	for (int i = 0; i < contours.size(); i++)
	//	{
	//		int flag = 0;

	//		//cv::drawContours(drawing1, approx, i, cv::Scalar::all(255), 1);
	//		for (int x = 0; x < approx[i].size(); x++) {
	//			flag++;
	//			//putText(drawing2, std::to_string(flag), Point(approx[i][x].x, approx[i][x].y), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);
	//		}
	//		putText(drawing2, std::to_string(flag), Point(50, i * 50), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(255, 255, 255), 1);

	//	}
}

cv::Mat OpenCVImage::dotarea(cv::Mat src)
{
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	int flag = 0;

	Mat result;
	m_result.copyTo(result);

	Mat img, img_labels, img_depthChange, img_color, stats, centroids;

	double minVal1, maxVal1;
	minMaxLoc(src, &minVal1, &maxVal1);
	src.convertTo(img_depthChange, CV_8UC1);

	int nccomps = connectedComponentsWithStats(img_depthChange, img_labels, stats, centroids, 8);

	// nccomps 라벨의 집합 갯수 nccomps-1 이 라벨 집합의 갯수
	// 각 라벨의 첫 포인트만 담아서 영역 표시 

	vector<Vec3b> colors(nccomps + 1);
	colors[0] = Vec3b(0, 0, 0);
	for (int i = 1; i <= nccomps; i++)
	{
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		// if (stats.at<int>(i - 1, CC_STAT_AREA) < 100)
		// colors[i] = Vec3b(0, 0, 0);
	}
	img_color = Mat::zeros(img.size(), CV_8UC3);

	for (int y = 0; y < img_color.rows; y++)
		for (int x = 0; x < img_color.cols; x++)
		{
			int label = img_labels.at<int>(y, x);
			CV_Assert(0 <= label && label <= nccomps);
			img_color.at<Vec3b>(y, x) = colors[label];
		}

	cv::Rect dot_rect;
	Mat dot_gray, dot_mag, patch_template, sum_edge_tar, sum_edge_template;
	Scalar mean_src1, stdv_src1;
	float ncc_edge;

	for (int j = 1; j < nccomps; j++)
	{
		int area = stats.at<int>(j, CC_STAT_AREA);
		if (area > 1 && area < 1000)
		{
			int left = stats.at<int>(j, CC_STAT_LEFT);
			int top = stats.at<int>(j, CC_STAT_TOP);
			int width = stats.at<int>(j, CC_STAT_WIDTH);
			int height = stats.at<int>(j, CC_STAT_HEIGHT);

			int x = centroids.at<double>(j, 0);
			int y = centroids.at<double>(j, 1);

			int passcheck = 0;

			int rectSize = width * height;

			dot_rect.x = left;
			dot_rect.y = top;
			if (width < 2)
			{
				dot_rect.width = width;
			}
			else
			{
				dot_rect.x = left;// +((width / 2 / 2));
				dot_rect.width = width;// -((width / 2 / 2));
			}

			if (height < 2)
			{
				dot_rect.height = height;

			}
			else
			{
				dot_rect.y = top;// +((height / 2 / 2));
				dot_rect.height = height;// -((height / 2 / 2));
			}


			Mat convert_rect;
			m_tar_gray_obj(dot_rect).copyTo(dot_gray);
			src(dot_rect).copyTo(convert_rect);
			int pixelCnt = 0;
			int pixelSum = 0;
			int pixelMean = 0;


			uchar* convert_rect_ptr;
			uchar* dot_gray_ptr;

			vector<int> pixelData;
			double std = 0;

			for (int y = 0; y < dot_gray.rows; y++)
			{
				convert_rect_ptr = convert_rect.ptr<uchar>(y);
				dot_gray_ptr = dot_gray.ptr<uchar>(y);

				for (int x = 0; x < dot_gray.cols; x++)
				{

					if (convert_rect_ptr[x] == 0)
						dot_gray_ptr[x] = 0;
					else
					{
						pixelCnt++;
						pixelSum += (int)dot_gray_ptr[x];
						pixelData.push_back((int)dot_gray_ptr[x]);
					}


				}
			}


			if (pixelCnt != 0)
			{
				pixelMean = pixelSum / pixelCnt;

				float sum = 0;
				double var = 0;

				for (int i = 0; i < pixelData.size(); i++)
				{
					sum += pow(pixelData[i] - pixelMean, 2);
				}
				var = sum / pixelData.size();
				std = sqrt(var);
			}


			float sumEdge = 0;
			float meanEdge = 0;
			int edgePixelCnt = 0;

			m_edge_data(dot_rect).copyTo(dot_mag);
			float* edgeData = (float*)dot_mag.data;

			for (int y = 0; y < dot_mag.rows; y++)
			{
				convert_rect_ptr = convert_rect.ptr<uchar>(y);

				for (int x = 0; x < dot_mag.cols; x++)
				{
					if (convert_rect_ptr[x] == 0 && edgeData[y * dot_mag.cols + x] < 0.1)
						edgeData[y * dot_mag.cols + x] = 0;
					else if (edgeData[y * dot_mag.cols + x] >= 0.1 && convert_rect_ptr[x] != 0)
					{
						sumEdge += edgeData[y * dot_mag.cols + x];
						edgePixelCnt++;
					}
				}
			}

			if (edgePixelCnt != 0)
				meanEdge = sumEdge / edgePixelCnt;

			//	meanStdDev(dot_gray, mean_src1, stdv_src1);
			if (area > 2 && std > 2 && meanEdge > 0 && meanEdge < 0.6) {
				flag++;
				putText(result, to_string(area), cv::Point(left + 20, top - 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
				//putText(m_result, to_string(y), cv::Point(left + 20, top - 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
				putText(result, to_string(pixelMean), cv::Point(left + 20, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
				putText(result, to_string(meanEdge), cv::Point(left + 20, top - 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
				putText(result, to_string(std), cv::Point(left + 20, top - 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 200), 2);
				//(m_result, to_string(24), cv::Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
				//putText(m_result, to_string(x), cv::Point(left + 20, top - 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
				cv::rectangle(result, dot_rect, Scalar(255, 255, 255), 1);
			}

		}
	}
	return result;
}
