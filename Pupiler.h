//
//  Pupiler.h
//  iGlance
//
//  Created by Weihao Cheng on 13-11-14.
//

#ifndef __iGlance__Pupiler__
#define __iGlance__Pupiler__

#include <opencv2/core/core.hpp>
class Pupiler {
    
public:
    typedef struct Region {
        
        cv::Rect rect;
        int area;
        int tag;
        
    }Region;
    
    cv::Point pulCenter;
    cv::Rect pulRect;
    
    void procPupil(cv::Mat &img);
    void procEyeCorner(cv::Mat &img);
    
};



#endif 
