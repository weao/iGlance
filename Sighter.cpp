//
//  Sighter.cpp
//  iGlance
//
//  Created by Weihao Cheng on 13-11-11.
//

#include "Sighter.h"
#include <cmath>
#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void Sighter::loadLandMarkDetector(const char *filename) {


}


void Sighter::setEyeClassifier(MLClassifier *model) {

    eyeActM = model;

}

void Sighter::loadFaceDetector(const char *filename) {


    faceDetector.load(filename);
}

void Sighter::loadEyepairDetector(const char *filename) {
    eyepairDetector.load(filename);
}

void Sighter::loadEyeDetector(const char *filename) {
    eyeDetector.load(filename);
}



void Sighter::init() {

    sightState = STARE_NONE;

}

Sighter::~Sighter() {



}


void Sighter::procSight(cv::Mat &frame, cv::Mat &lfeat, cv::Mat &rfeat) {


    cv::cvtColor(frame, drawMat, CV_BGRA2BGR);
    cv::flip(drawMat, drawMat, 1);

    cv::cvtColor(drawMat, grayMat, CV_BGR2GRAY);

    cv::Rect frect;

    cv::Mat& draw = drawMat;
    cv::Mat& gray = grayMat;

    //Detect face
    cv::vector<cv::Rect> faces;
    faceDetector.detectMultiScale(gray, faces, 1.1, 20, CV_HAAR_DO_CANNY_PRUNING|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(gray.cols/4,gray.rows/4));

    if(faces.empty()) {
        return;
    }

    frect = faces[0];

    cv::Mat face = gray(frect);

    //Detect eye
    cv::vector<cv::Rect> leyes,reyes;

    cv::Size max_size = cv::Size(face.cols/2,face.rows/4);
    cv::Size min_size = cv::Size(face.cols/10, 10);

    cv::Mat lface = face(cv::Rect(0,face.rows/4,face.cols/2,face.rows/3));
    eyeDetector.detectMultiScale(lface, leyes, 1.1, 20, CV_HAAR_DO_CANNY_PRUNING, min_size, max_size);

    cv::Mat rface = face(cv::Rect(face.cols/2,face.rows/4,face.cols/2,face.rows/3));
    eyeDetector.detectMultiScale(rface, reyes, 1.1, 20, CV_HAAR_DO_CANNY_PRUNING, min_size, max_size);

    int szl = (int)leyes.size();
    int szr = (int)reyes.size();

    if(szl < 1 || szr < 1) {
        return;
    }


    //have min vertical difference
    int min_dy = INT_MAX;
    cv::Rect lerect, rerect;
    for(int i=0;i<szl;i++) {

        int cy = leyes[i].y + leyes[i].height/2;

        for(int j=0;j<szr;j++) {

            int cyr = reyes[j].y + reyes[j].height/2;

            int d = abs(cy-cyr);

            if( d < min_dy) {
                min_dy = d;
                lerect = leyes[i];
                rerect = reyes[j];
            }
        }
    }

    //transform to worldwide coordinate
    lerect.x += frect.x;
    lerect.y += frect.y + frect.height/4;

    rerect.x += frect.x + frect.width/2;
    rerect.y += frect.y + frect.height/4;


    //classifing eye's action
    cv::Mat legray = gray(lerect);
    cv::Mat regray = gray(rerect);

    cv::flip(regray, regray, 1);

    lfeat = legray;
    rfeat = regray;

    cv::Mat sample(50,50,CV_8UC1);

    cv::resize(legray, sample, sample.size());


    int ldir = -1;
    if(eyeActM)
        ldir = eyeActM->predict(sample);

    cv::resize(regray, sample, sample.size());

    int rdir = -1;
    if(eyeActM)
        rdir = eyeActM->predict(sample);


    SightState sstate = STARE_NONE;
    if( ldir == 0 && rdir == 0 ) {
        sstate = STARE_CENTER;
    }
    if( ldir == 2 && rdir == 1) {
        sstate = STARE_LEFT;

    }
    if( ldir == 1 && rdir == 2) {
        sstate = STARE_RIGHT;
    }

    {
#define ACT_STAT_TOTAL 10

        static SightState sstates[ACT_STAT_TOTAL] = {STARE_NONE};
        static int ssi = 0;
        static int ssleft = 0;
        static int ssright = 0;
        static int sscenter = 0;
        static int ssnone = ACT_STAT_TOTAL;

        switch(sstates[ssi]) {
            case STARE_NONE:
                ssnone--;
                break;
            case STARE_LEFT:
                ssleft--;
                break;
            case STARE_RIGHT:
                ssright--;
                break;
            case STARE_CENTER:
                sscenter--;
                break;
            default:
                break;
        }
        switch(sstate) {
            case STARE_NONE:
                ssnone++;
                if(ssnone >= ACT_STAT_TOTAL*0.6) {
                    sightState = STARE_NONE;
                }
                break;
            case STARE_LEFT:
                ssleft++;
                if(ssleft >= ACT_STAT_TOTAL*0.6) {
                    sightState = STARE_LEFT;
                }
                break;
            case STARE_RIGHT:
                ssright++;
                if(ssright >= ACT_STAT_TOTAL*0.6) {
                    sightState = STARE_RIGHT;
                }

                break;
            case STARE_CENTER:
                sscenter++;
                if(sscenter >= ACT_STAT_TOTAL*0.6) {
                    sightState = STARE_CENTER;
                }
                break;
            default:
                break;
        }

        sstates[ssi] = sstate;
        ssi = (ssi+1)%ACT_STAT_TOTAL;


    }


    //tracking the pupil
    pulLeft.procPupil(legray);
    pulRight.procPupil(regray);



    pulLeft.pulCenter += lerect.tl();
    pulRight.pulCenter += rerect.tl();



    //Drawing

    int r = MAX(pulLeft.pulRect.height, pulRight.pulRect.height)/3;

    static int ani_r = 100;
    static int ani_speed = 5;


    if(ani_r > 0) {


        cv::circle(draw, pulLeft.pulCenter, ani_r, CV_RGB(255,0,0),1.5);
        cv::circle(draw, pulRight.pulCenter, ani_r, CV_RGB(255,0,0),1.5);
        ani_r -= ani_speed;
        ani_speed += 5;

    }
    else if( r > 0 ) {



        cv::circle(draw, pulLeft.pulCenter, r, CV_RGB(255,0,0), CV_FILLED);
        cv::circle(draw, pulLeft.pulCenter, r*3, CV_RGB(255,0,0), 1);

        cv::circle(draw, pulRight.pulCenter, r, CV_RGB(255,0,0), CV_FILLED);
        cv::circle(draw, pulRight.pulCenter, r*3, CV_RGB(255,0,0), 1);


    }

    cv::rectangle(draw, frect, CV_RGB(255,255,0));
    cv::rectangle(draw, lerect, CV_RGB(0,255,0));
    cv::rectangle(draw, rerect, CV_RGB(0,255,0));
}
