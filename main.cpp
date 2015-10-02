//
//  main.cpp
//  iGlance
//
//  Created by Weihao Cheng on 13-12-7.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Sighter.h"
#include "MLClassifier.h"

int main(int argc, const char * argv[])
{

    cv::VideoCapture camera;

    camera.open(CV_CAP_ANY);

    if(!camera.isOpened()) {

        std::cout<<"No camera detected!";
        return 0;
    }


    Sighter *sighter = new Sighter;
    sighter->init();
    sighter->loadFaceDetector("./resource/haarcascade_frontalface_alt.xml");
    sighter->loadEyeDetector("./resource/haarcascade_eye.xml");

    MLClassifier mlc(MLClassifier::SVM);
    mlc.loadPCA("./resource/pca.dat");
    mlc.loadModel("./resource/svm.dat");

    sighter->setEyeClassifier(&mlc);


    cv::Mat frame, lfeature, rfeature;

    Sighter::SightState prev_state = Sighter::STARE_NONE;

    while(camera.read(frame)) {

        sighter->procSight(frame, lfeature, rfeature);

        //Action happends when state changes
        if(sighter->sightState != prev_state) {

            prev_state = sighter->sightState;
            switch(prev_state) {
                case Sighter::STARE_LEFT:
                    std::cout<<"Stare left"<<std::endl;
                    break;
                case Sighter::STARE_RIGHT:
                    std::cout<<"Stare right"<<std::endl;
                    break;
                default:
                    break;
            }
        }

        cv::imshow("output", sighter->drawMat);

        char key = cv::waitKey(1);

        //Press 'ESC' to exit
        if(key == 27) {
            break;
        }
    }

    delete sighter;

    return 0;
}
