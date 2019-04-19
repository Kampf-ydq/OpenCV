
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
	CvBox2D Armor;//����װ�׵ľ�������
	int nL, nW;
	vRlt.clear();
	if (vEllipse.size() < 2)//�����⵽����ת���θ���<2,ֱ�ӷ���
		return vRlt;
	for (unsigned int nI = 0; nI < vEllipse.size() - 1; nI++)
	{
		for (unsigned int nJ = nI + 1; nJ < vEllipse.size(); nJ++)
		{
			//�ж���������ת�����Ƿ���һ��װ�׵�����LED����
			if (abs(vEllipse[nI].angle - vEllipse[nJ].angle) < T_ANGLE_THRE && abs(vEllipse[nI].size.height - vEllipse[nJ].size.height) < (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 10 && abs(vEllipse[nI].size.width - vEllipse[nJ].size.width) < (vEllipse[nI].size.width + vEllipse[nJ].size.width) / 10)
			{
				Armor.center.x = (vEllipse[nI].center.x + vEllipse[nJ].center.x) / 2;//װ������x
				Armor.center.y = (vEllipse[nI].center.y + vEllipse[nJ].center.y) / 2;//װ������y
				Armor.angle = (vEllipse[nI].angle + vEllipse[nJ].angle) / 2;//װ����ת�Ƕ�
				nL = (vEllipse[nI].size.height + vEllipse[nJ].size.height) / 2;//װ�׵ĸ߶�

																			   //װ�׵Ŀ�� = center1(x1,y1) center2(x2,y2)�����ľ���
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
				vRlt.push_back(Armor);//���ҳ���װ�׵���ת���α��浽vector
			}
		}
	}
	return vRlt;
}

void DrawBox(CvBox2D box, IplImage* img)
{
	CvPoint2D32f point[4];//��ά�����µĵ㣬����Ϊ���� 
	int i;

	//��ʼ��
	for (i = 0; i<4; i++)
	{
		point[i].x = 0;
		point[i].y = 0;
	}
	cvBoxPoints(box, point); //�����ά���Ӷ��� 
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
	CvCapture* pCapture0 = cvCreateFileCapture("RawImage\\BlueCar.avi");//cvCreateFileCapture()ͨ����������ȷ��Ҫ�����avi�ļ�
																		//CvCapture��һ���ṹ�壬�������沶�񵽵�ͼ����Ϣ
																		//CvCapture* pCapture0 = cvCreateCameraCapture(0);
	IplImage* pFrame0 = NULL;
	CvSize pImgSize;
	CvScalar sColour;			//****************CvBox2D(RotatedRect)��������********************//
	CvBox2D s;					//CvBox2D��3����Ա������center(������x,y),size(��͸�),angle(��ת��)//			
	vector<CvBox2D> vEllipse;	//			����angleΪ��ת��������ֱ����ļн�					 //
	vector<CvBox2D> vRlt;
	vector<CvBox2D> vArmor;
	CvScalar sl;
	bool bFlag = false;//�ж��Ƿ��⵽Ŀ������
	CvSeq *pContour = NULL;//��������һ��һ�������ص���ɵģ����Կ���CvSeq�����У����洢

	pFrame0 = cvQueryFrame(pCapture0);//cvQueryFrame��ʾ������ͷ�����ļ�ץȡ������һ֡������Ƶ�ļ������ڴ�

	pImgSize = cvGetSize(pFrame0);

	//����һ��RGBͼ��
	IplImage *pRawImg = cvCreateImage(pImgSize, IPL_DEPTH_8U, 3);//����cvCreateImage����ͼ���׵�ַ��������洢�ռ䡣
																 //IplImage* cvCreateImage(CvSize cvSize(int width, int height), int depth, int channels);
																 //�����Ҷ�ͼ
	IplImage* pHImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pRImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage* pGImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBImage = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pBinary = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);
	IplImage *pRlt = cvCreateImage(pImgSize, IPL_DEPTH_8U, 1);

	CvSeq* lines = NULL;

	//CvMemStorage* storage = cvCreateMemStorage(block_size);
	//��������һ���ڴ�洢������ͳһ������ֶ�̬������ڴ档
	//��������һ���´������ڴ�洢��ָ�롣
	//����block_size��Ӧ�ڴ�����ÿ���ڴ��Ĵ�С��Ϊ0ʱ�ڴ��Ĭ�ϴ�СΪ64k

	CvMemStorage* storage = cvCreateMemStorage(0);
	CvMemStorage* pStorage = cvCreateMemStorage(0);
	while (1)
	{
		if (pFrame0)
		{
			cvSplit(pFrame0, pBImage, pGImage, pRImage, 0);//����R��G��B����ͨ��
			GetDiffImage(pBImage, pRImage, pBinary, 90);
			cvDilate(pBinary, pHImage, NULL, 3);
			cvErode(pHImage, pRlt, NULL, 1);

			//����cvFindContours�Ӷ�ֵͼ���м��������������ؼ�⵽�������ĸ�����
			//first_contour��ֵ�ɺ�����䷵�أ�����ֵ��Ϊ��һ����������ָ�룬��û����������⵽ʱΪNULL��������������ʹ��h_next��v_next���ӣ���first_contour���

			cvFindContours(pRlt, pStorage, &pContour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
			for (; pContour != NULL; pContour = pContour->h_next)
			{
				if (pContour->total > 10)//�жϵ�ǰ�����Ƿ����10�����ص�
				{
					bFlag = true;//�ǣ����⵽Ŀ������

					s = cvFitEllipse2(pContour);//���Ŀ�������Ϊ��Բ������һ����ת���Σ����ġ��Ƕȡ��ߴ磩

												//��������ת�������ĵ�Ϊ���ĵ�5*5�����ؿ�
												//��ע�⡿Ҫ�Ӷ�ֵ��֮ǰ��ͼ��
					for (int nI = 0; nI < 5; nI++)
					{
						for (int nJ = 0; nJ < 5; nJ++)
						{
							if (s.center.y - 2 + nJ > 0 && s.center.y - 2 + nJ < 480 && s.center.x - 2 + nI > 0 && s.center.x - 2 + nI <  640)//�ж������Ƿ�����Чλ��
							{
								sl = cvGet2D(pFrame0, (int)(s.center.y - 2 + nJ), (int)(s.center.x - 2 + nI));//������ȡ���ص������ֵ
								if (sl.val[0] < 200 || sl.val[1] < 200 || sl.val[2] < 200)//�ж����ĵ��Ƿ�ӽ���ɫ
									bFlag = false;
							}
						}
					}
					if (bFlag)
					{
						vEllipse.push_back(s);//�����ֵ�Ŀ�걣��
											  //cvEllipseBox(pFrame0, s, CV_RGB(255, 0, 0), 2, 8, 0);
					}
				}

			}

			//�����ӳ����������LED������ת���ε�vector���ҳ�װ�׵�λ�ã�����װ����ת���Σ�����vector������
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
