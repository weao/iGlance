//
//  MLClassifier.h
//  iGlance
//
//  Created by Weihao Cheng on 13-12-6.
//

#ifndef __MLClassifier__
#define __MLClassifier__

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

class MLClassifier {

    cv::Mat _mean;
    cv::Mat _eigenvectors; //by columns

    CvStatModel *_model;

    int _type;

public:
    enum {
        NormalBayes = 0,
        SVM = 1,
        DTree = 2,
        NerualNetwork = 3
    };

    MLClassifier() {
        _model = NULL;
    }

    MLClassifier(int type) {
        _model = NULL;
        init(type);
    }

    ~MLClassifier() {

        _model->clear();
        delete _model;
    }

    void init(int type) {

        if(_model) {
            _model->clear();
            delete _model;
        }

        switch (type) {
            case NormalBayes:
                _model = new CvNormalBayesClassifier;
                break;
            case SVM:
                _model = new CvSVM;
                break;
            case DTree:
                _model = new CvDTree;
                break;
            case NerualNetwork:
                _model = new CvANN_MLP;
                break;

            default:
                _model = NULL;
                break;
        }
        _type = type;

    }

    void train(cv::Mat &data, cv::Mat_<int> &labels, int components);


    float predict(cv::Mat &sample);

    void savePCA(cv::string filename);
    void loadPCA(cv::string filename);
    void saveModel(cv::string filename);
    void loadModel(cv::string filename);



};

#endif
