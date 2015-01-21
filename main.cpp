#include <opencv2\opencv.hpp>
#include "utils.h"
#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
using namespace std;
using namespace cv;


int main(int argc, char** argv){
	const int DIR_LENGTH = 256;
	const int DIR_NUM = 40;
	const char *dir[1] = { "E:\\BaiduYunDownload\\Useful data\\ORL database\\" };
	char path[DIR_LENGTH];
	IplImage* img;
	IplImage *change;
	vector<cv::Mat> InImgs;
	cv::Mat* bmtx;
	cv::Mat* histo; // histogram equalizaion  

	const int train_num = 5;
	const int NUM = DIR_NUM * train_num;

	float *labels = new float[NUM];
	int x = 0;
	for (int i = 1; i <= DIR_NUM; i++){
		for (int j = 1; j <= train_num; j++){
			sprintf(path, "%s%c%d%s%d%s", dir[0], 's', i, "\\", j, ".bmp");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);

			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0 / 255, 0);
			bmtx = new cv::Mat(change);
			InImgs.push_back(*bmtx);
			labels[x] = (float)i;
			x++;
		}
	}


	vector<int> NumFilters;
	NumFilters.push_back(8);
	NumFilters.push_back(8);
	vector<int> blockSize;
	blockSize.push_back(28);
	blockSize.push_back(23);

	PCANet pcaNet = {
		2,
		7,
		NumFilters,
		blockSize,
		0.5
	};

	cout << "\n ====== PCANet Training ======= \n" << endl;
	int64 e1 = cv::getTickCount();
	PCA_Train_Result* result = PCANet_train(InImgs, &pcaNet, true);
	int64 e2 = cv::getTickCount();
	double time = (e2 - e1) / cv::getTickFrequency();
	cout << " PCANet Training time: " << time << endl;


	FileStorage fs("E:\\BaiduYunDownload\\Useful data\\ORL database\\filters_60x48_2.xml", FileStorage::WRITE);
	fs << "filter1" << result->Filters[0] << "filter2" << result->Filters[1];
	fs.release();

	///  svm  train  //////////
	cout << "\n ====== Training Linear SVM Classifier ======= \n" << endl;

	float *new_labels = new float[NUM];
	int size = result->feature_idx.size();
	for (int i = 0; i < size; i++)
		new_labels[i] = labels[result->feature_idx[i]];



	Mat labelsMat(NUM, 1, CV_32FC1, new_labels);

	result->Features.convertTo(result->Features, CV_32F);

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.C = 1;
	//params.nu = 0.8;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	CvSVM SVM;

	e1 = cv::getTickCount();
	SVM.train(result->Features, labelsMat, Mat(), Mat(), params);
	///  svm  train    /////
	e2 = cv::getTickCount();
	time = (e2 - e1) / cv::getTickFrequency();
	cout << " svm training complete, time usage: " << time << endl;

	SVM.save("E:\\BaiduYunDownload\\Useful data\\ORL database\\svm_60x48_2.xml");

	result->Features.deallocate();


	cout << "\n ====== PCANet Testing ======= \n" << endl;

	vector<Mat> testImg;
	vector<int> testLabel;
	vector<string> names;
	string *t;

	int testNum = 5;

	for (int i = 1; i <= DIR_NUM; i++){
		for (int j = train_num + 1; j <= train_num + testNum; j++){
			sprintf(path, "%s%c%d%s%d%s", dir[0], 's', i, "\\", j, ".bmp");
			img = cvLoadImage(path, CV_LOAD_IMAGE_GRAYSCALE);
			t = new string(path);
			names.push_back(*t);
			change = cvCreateImage(cvGetSize(img), IPL_DEPTH_64F, img->nChannels);
			cvConvertScale(img, change, 1.0 / 255, 0);
			bmtx = new cv::Mat(change);
			testImg.push_back(*bmtx);
			//testLabel.push_back(1);
			testLabel.push_back(i);
		}
	}
	int testSIze = testImg.size();
	Hashing_Result* hashing_r;
	PCA_Out_Result *out;

	float all = DIR_NUM * testNum;
	float correct = 0;
	int coreNum = omp_get_num_procs();

	float *corrs = new float[DIR_NUM];
	for (int i = 0; i < DIR_NUM; i++)
		corrs[i] = 0;



	e1 = cv::getTickCount();
# pragma omp parallel for default(none) num_threads(coreNum) private(out, hashing_r) shared(names, corrs, correct, testLabel, SVM, pcaNet, testSIze, testImg, result, testNum)
	for (int i = 0; i < testSIze; i++){
		out = new PCA_Out_Result;
		out->OutImgIdx.push_back(0);
		out->OutImg.push_back(testImg[i]);
		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize,
			pcaNet.NumFilters[0], result->Filters[0], 2);
		for (int j = 1; j < pcaNet.NumFilters[1]; j++)
			out->OutImgIdx.push_back(j);

		out = PCA_output(out->OutImg, out->OutImgIdx, pcaNet.PatchSize,
			pcaNet.NumFilters[1], result->Filters[1], 2);
		hashing_r = HashingHist(&pcaNet, out->OutImgIdx, out->OutImg);
		hashing_r->Features.convertTo(hashing_r->Features, CV_32F);
		int pred = SVM.predict(hashing_r->Features);
#pragma omp critical 
		{
			printf("predict: %d, testLabel: %d\n", pred, testLabel[i]);
			if (pred == testLabel[i]){
				corrs[testLabel[i] - 1]++;
				correct++;
			}
			else printf(" pred: %d , label:%d \n", pred, testLabel[i]);
		}
		delete out;
	}



	e2 = cv::getTickCount();
	time = (e2 - e1)/ cv::getTickFrequency();
	cout <<" test time usage: "<<time<<endl;
	cout <<"all precise: "<<correct / all<<endl;
	for(int i=0; i<DIR_NUM; i++)
		cout <<"individual"<<i+1<<" precise: "<<corrs[i] / testNum<<endl;
	cout <<"test images num for each class: "<<testNum<<endl;
	
	getchar();
	return 0;
}
