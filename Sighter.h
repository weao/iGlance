//
//  Sighter.h
//  iGlance
//
//  Created by Weihao Cheng on 13-11-11.
//

#ifndef __iGlance__Sighter__
#define __iGlance__Sighter__


#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "Pupiler.h"
#include "MLClassifier.h"

class Sighter {
    
public:
    Sighter() {
        sightState = STARE_NONE;
        eyeActM = NULL;

    }
    ~Sighter();
    

    typedef enum {
        
        STARE_NONE = 0,
        STARE_CENTER = 1,
        STARE_LEFT = 2,
        STARE_RIGHT = 3
    }SightState;
    
    
    //result of eye's action: STARE_NONE, STARE_LEFT, STARE_RIGHT
    SightState sightState;

    //classify eye's action
    MLClassifier *eyeActM;
    
    Pupiler pulLeft;
    Pupiler pulRight;


    cv::Mat drawMat;
    cv::Mat grayMat;
    
    
    cv::CascadeClassifier faceDetector;
    cv::CascadeClassifier eyeDetector;
    cv::CascadeClassifier eyepairDetector;
    
    void setEyeClassifier(MLClassifier *model);
    void loadLandMarkDetector(const char *filename);
    void loadFaceDetector(const char *filename);
    void loadEyepairDetector(const char *filename);
    
    void loadEyeDetector(const char *filename);

    void init();
    void procSight(cv::Mat &frame, cv::Mat &lfeat, cv::Mat &rfeat);
    void preprocSight(cv::Mat &frame);
    
};

#endif 
