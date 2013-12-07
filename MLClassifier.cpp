//
//  SVMClassifier.cpp
//  iGlance
//
//  Created by Weihao Cheng on 13-12-6.
//

#include "MLClassifier.h"

#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>


void MLClassifier::train(cv::Mat &data, cv::Mat_<int> &labels, int components) {
    
    cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, components);
    
    _mean = pca.mean.reshape(1,1);
    
    _eigenvectors = pca.eigenvectors.t();
    
    cv::Mat fdata = cv::subspaceProject(_eigenvectors, _mean, data);
    
    
    if(_type == SVM) {
        
        //SVM params setting
        
        CvSVMParams params;
        params.svm_type = cv::SVM::C_SVC;
        params.kernel_type = cv::SVM::POLY;
        params.degree = 3;
        params.coef0 = 1;
        
        CvSVM *svm = (CvSVM*)_model;
        svm->train(fdata, labels, cv::Mat(), cv::Mat(), params);
    }
    else if(_type == NormalBayes) {
        
        CvNormalBayesClassifier *nbc = (CvNormalBayesClassifier*)_model;
        nbc->train(fdata, labels);
    
    }
    
    
}


float MLClassifier::predict(cv::Mat &sample) {
    
    
    cv::Mat feature = cv::subspaceProject(_eigenvectors, _mean, sample.reshape(1,1));
    
    if(_type == NormalBayes) {
        CvNormalBayesClassifier *nbc = (CvNormalBayesClassifier*)_model;
        return nbc->predict(feature);
    }
    else if(_type == SVM) {
        CvSVM *svm = (CvSVM*)_model;
        return svm->predict(feature);
        
    }
    
    return -1;
    
    
}

void MLClassifier::savePCA(cv::string path) {
    
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if(fs.isOpened()) {
        fs << "eigenvectors" << _eigenvectors;
        fs << "mean" << _mean;
    }
    
}

void MLClassifier::saveModel(cv::string path) {
    
    _model->save(path.c_str());
    
}

void MLClassifier::loadPCA(cv::string path) {
    
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if(fs.isOpened()) {
        fs["eigenvectors"] >> _eigenvectors;
        fs["mean"] >> _mean;
    }
}

void MLClassifier::loadModel(cv::string path) {
    
    _model->load(path.c_str());
    
}
