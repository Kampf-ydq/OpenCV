
//#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include "cxcore.h"
#include "omp.h"

using namespace cv;
using namespace std;

#define T_ANGLE_THRE 10
#define T_SIZE_THRE 10

//							B				R				Binary    90
void GetDiffImage(IplImage* src1, IplImage* src2, IplImage* dst, int nThre)
{
	unsigned char* SrcData1 = (unsigned char*)src1->imageData;
	unsigned char* SrcData2 = (unsigned char*)src2->imageData;
	unsigned char* DstData = (unsigned char*)dst->imageData;
	int step = src1->widthStep / sizeof(unsigned char);

	omp_set_num_threads(8);
#pragma omp parallel for

	for (int nI = 0; nI<src1->height; nI++)
	{
		for (int nJ = 0; nJ <src1->width; nJ++)
		{
			if (SrcData1[nI*step + nJ] - SrcData2[nI*step + nJ]> nThre)
			{
				DstData[nI*step + nJ] = 255;
			}
			else
			{
				DstData[nI*step + nJ] = 0;
			}
		}
	}
}

vector<CvBox2D> ArmorDetect(vector<CvBox2D> vEllipse)
{
	vector<CvBox2D> vRlt;
	CvBox2D Armor;//定义装甲的矩形区域
	int nL, nW;
	vRlt.clear();
	if (vEllipse.size() < 2)//如果检测到的旋转矩形个数<2,直接返回
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++)
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			//判断这两个旋转矩形是否是一个装甲的两个LED等条
			if (abs(vEllipse[nI].angle - vEllipse[nJ].angle) < T_ANGLE_THRE && abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 10 && abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / 10)
			{
				Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2;//装甲中心x
				Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2;//装甲中心y
				Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;//装甲旋转角度
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2;//装甲的高度

																			   //装甲的宽度 = center1(x1,y1) center2(x2,y2)两点间的距离
				nW = sqrt((vEllipse[nI].center.x - vEllipse[nJ].center.x) * (vEllipse[nI].center.x - vEllipse[nJ].center.x) + (vEllipse[nI].center.y - vEllipse[nJ].center.y) * (vEllipse[nI].center.y - vEllipse[nJ].center.y));
				if (nL < nW)
				{
					Armor.size.height = nL;
					Armor.size.width = nW;
				}
				else
				{
					Armor.size.height = nW;
					Armor.size.width = nL;
				}
				vRlt.push_back(Armor);//将找出的装甲的旋转矩形保存到vector
			}
		}
	}
	return vRlt;
}

void DrawBox(CvBox2D box, IplImage* img)
{
	CvPoint2D32f point[4];//二维坐标下的点，类型为浮点 
	int i;

	//初始化
	for (i = 0; i<4; i++)
	{
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point); //计算二维盒子顶点 
	CvPoint pt[4];
	for (i = 0; i<4; i++)
	{
		pt[i].x = (int)point[i].x;
		pt[i].y = (int)point[i].y;
	}
	cvLine(img, pt[0], pt[1], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[1], pt[2], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[2], pt[3], CV_RGB(0, 0, 255), 2, 8, 0);
	cvLine(img, pt[3], pt[0], CV_RGB(0, 0, 255), 2, 8, 0);
}

int main()
{
	CvCapture* pCapture0 = cvCreateFileCapture("RawImage\\BlueCar.avi");//cvCreateFileCapture()通过参数设置确定要读入的avi文件
																		//CvCapture是一个结构体，用来保存捕获到的图像信息
																		//CvCapture* pCapture0 = cvCreateCameraCapture(0);
	IplImage* pFrame0 = NULL;
	CvSize pImgSize;
	CvScalar sColour;			//****************CvBox2D(RotatedRect)数据类型********************//
	CvBox2D s;					//CvBox2D有3个成员变量，center(块中心x,y),size(宽和高),angle(旋转角)//			
	vector<CvBox2D> vEllipse;	//			其中angle为旋转矩形与竖直方向的夹角					 //
	vector<CvBox2D> vRlt;
	vector<CvBox2D> vArmor;
	CvScalar sl;
	bool bFlag = false;//判断是否检测到目标区域
	CvSeq *pContour = NULL;//轮廓是由一个一个的像素点组成的，所以可用CvSeq（序列）来存储

	pFrame0 = cvQueryFrame(pCapture0);//cvQueryFrame表示从摄像头或者文件抓取并返回一帧，将视频文件载入内存

	pImgSize = cvGetSize(pFrame0);

	//创建一张RGB图像
	IplImage *pRawImg = cvCreateImage(pImgSize, IPL_DEPTH_8U, 3);//函数cvCreateImage创建图像首地址，并分配存储空间。
																 //IplImage* cvCreateImage(CvSize cvSize(int width, int height), int depth, int channels);
																 //创建灰度图
	IplImage* pHImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pRImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pGImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBinary = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pRlt = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);

	CvSeq* lines = NULL;

	//CvMemStorage* storage = cvCreateMemStorage(block_size);
	//用来创建一个内存存储器，来统一管理各种动态对象的内存。
	//函数返回一个新创建的内存存储器指针。
	//参数block_size对应内存器中每个内存块的大小，为0时内存块默认大小为64k

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvMemStorage* pStorage = cvCreateMemStorage(0);
	while (1)
	{
		if (pFrame0)
		{
			cvSplit(pFrame0, pBImage, pGImage, pRImage, 0);//分离R、G、B三个通道
			GetDiffImage(pBImage, pRImage, pBinary, 90);
			cvDilate(pBinary, pHImage, NULL, 3);
			cvErode(pHImage, pRlt, NULL, 1);

			//函数cvFindContours从二值图像中检索轮廓，并返回检测到的轮廓的个数。
			//first_contour的值由函数填充返回，它的值将为第一个外轮廓的指针，当没有轮廓被检测到时为NULL。其它轮廓可以使用h_next和v_next连接，从first_contour到达。

			cvFindContours(pRlt, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			for (; pContour != NULL; pContour = pContour->h_next)
			{
				if (pContour->total > 10)//判断当前轮廓是否大于10个像素点
				{
					bFlag = true;//是，则检测到目标区域

					s = cvFitEllipse2(pContour);//拟合目标区域成为椭圆，返回一个旋转矩形（中心、角度、尺寸）

												//遍历以旋转矩形中心点为中心的5*5的像素块
												//【注意】要从二值化之前的图像看
					for (int nI = 0; nI < 5; nI++)
					{
						for (int nJ = 0; nJ < 5; nJ++)
						{
							if (s.center.y - 2 + nJ > 0 && s.center.y - 2 + nJ < 480 && s.center.x - 2 + nI > 0 && s.center.x - 2 + nI <  640)//判断像素是否在有效位置
							{
								sl = cvGet2D(pFrame0, (int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI));//遍历获取像素点的像素值
								if (sl.val[0] < 200 || sl.val[1] < 200 || sl.val[2] < 200)//判断中心点是否接近白色
									bFlag = false;
							}
						}
					}
					if (bFlag)
					{
						vEllipse.push_back(s);//将发现的目标保存
											  //cvEllipseBox(pFrame0, s, CV_RGB(255, 0, 0), 2, 8, 0);
					}
				}

			}

			//调用子程序，在输入的LED所在旋转矩形的vector中找出装甲的位置，并包装成旋转矩形，存入vector并返回
			vRlt = ArmorDetect(vEllipse);

			for (unsigned int nI = 0; nI < vRlt.size(); nI++)
				DrawBox(vRlt[nI], pFrame0);


			//cvWriteFrame(writer, pRawImg);
			cvShowImage("Raw", pFrame0);
			cvWaitKey(0);
			vEllipse.clear();
			vRlt.clear();
			vArmor.clear();
		}
		pFrame0 = cvQueryFrame(pCapture0);
	}
	cvReleaseCapture(&pCapture0);
	return 0;
}
