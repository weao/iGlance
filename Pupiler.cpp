//
//  Pupiler.cpp
//  iGlance
//
//  Created by Weihao Cheng on 13-11-14.
//

#include <opencv2/imgproc/imgproc.hpp>
#include "Pupiler.h"
#include <cmath>

static cv::Point getMaxCircleCenter(cv::Mat &sub, cv::vector< cv::vector<cv::Point> > &contours) {
    
    const int N = 3;
    float max_dis = 0;
    cv::Point center = cvPoint(-1, -1);
    
    
    //int sz = (int)contour.size();
    
    const uchar *ptr = sub.ptr();
    
    for(int y=0;y<sub.rows;y+=N) {
        
        for(int x=0;x<sub.cols;x+=N) {
            
            if(ptr[x] > 0) {
                
                cv::Point test = cv::Point(x,y);
                float min_dis = FLT_MAX;
                
                for(int j=0,je=(int)contours.size();j<je && min_dis > max_dis;j++) {
                    
                    cv::vector<cv::Point> &c = contours[j];
                    
                    for(int i=0,e=(int)c.size();i<e && min_dis > max_dis;i++) {
                        
                        float d = powf(c[i].x-test.x,2) + powf(c[i].y-test.y,2);
                        if(d < min_dis) {
                            min_dis = d;
                        }
                    }
                }
                if(min_dis > max_dis) {
                    
                    max_dis = min_dis;
                    center = test;
                }
                
            }
            
        }
        ptr += N*sub.step[0];
    
        
    }
    return center;
    
}

static void findContourRegion(cv::Mat &src, cv::Mat &dst, int aperture_size) {
    
    cv::Mat dx(src.rows, src.cols, CV_16SC(1));
    cv::Mat dy(src.rows, src.cols, CV_16SC(1));
    
    cv::Mat temp(src.rows, src.cols, CV_32FC1);
    
    
    cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    
    
    //mcy
    
    int i=0;
    int j=0;
    
    
    for(i = 0; i < src.rows; i++ )
    {
        const short* _dx = (short*)(dx.data + dx.step*i);
        const short* _dy = (short*)(dy.data + dy.step*i);
        float* _image = (float *)(temp.data + temp.step*i);
        
        for(j = 0; j < src.cols; j++)
        {
            _image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
        }
    }
    
    cv::convertScaleAbs(temp, dst);
    
}






void Pupiler::procPupil(cv::Mat &img) {
    
    
      
    cv::Mat &gray = img;//(img.rows,img.cols,CV_8UC1);
    
    //cv::cvtColor(img, gray, CV_BGR2GRAY);
    
    int bin[256] = {0};
    uchar *ptr = gray.ptr();
    
    for(int r=0;r<gray.rows;r++) {
        
        for(int c=0;c<gray.cols;c++) {
            
            bin[ptr[c]]++;
        }
        ptr += gray.step[0];
    }
    
    int sz = img.rows * img.cols;
    int thres = 1;
    int acc = bin[0];
    for(thres=1;thres<256;thres++) {
        
        if(acc > sz/10) {
            break;
        }
        acc += bin[thres];
        
    }
    
    cv::threshold(gray, gray, thres, 100, CV_THRESH_BINARY_INV);

    
    
    int tag = 0;
        
    cv::vector<Pupiler::Region> vec;
    
    ptr = gray.ptr();
    for(int r=0;r<gray.rows;r++) {
        
        for(int c=0;c<gray.cols;c++) {
            
            if(ptr[c] == 100) {
                
                Pupiler::Region region;
                cv::Rect rect;
                int area = cv::floodFill(gray, cv::Point(c,r), 120+tag, &rect);
                
                region.area = area;
                region.rect = rect;
                region.tag = tag;
                
                
                vec.push_back(region);
                
                tag++;
            }
            
        }
        ptr += gray.step[0];
    }
    
    
    int vsz = (int)vec.size();
    
    float max_area = FLT_MIN;
    float min_area = FLT_MAX;
    float max_dist = FLT_MIN;
    float min_dist = FLT_MAX;
    
    cv::vector< std::pair<float,float> > ranks(vsz);
    
    for(int i=0;i<vsz;i++) {
        
        if(vec[i].area > max_area) {
            max_area = vec[i].area;
        }
        if(vec[i].area < min_area) {
            min_area = vec[i].area;
        }

        
        cv::Point c;
        c.x = vec[i].rect.x + vec[i].rect.width/2;
        c.y = vec[i].rect.y + vec[i].rect.height/2;
        
        float dist = sqrtf(powf(c.x-gray.cols/2,2)+powf(c.y-gray.rows/2,2));
        if(dist > max_dist) {
            max_dist = dist;
        }
        else if(dist < min_dist) {
            min_dist = dist;
        }
        
        ranks[i].first = vec[i].area;
        ranks[i].second = dist;
    }
    
    float max_score = FLT_MIN;
    int ri = 0;
    for(int i=0;i<vsz;i++) {
        
        float score = (ranks[i].first - min_area) / (max_area - min_area) - (ranks[i].second - min_dist) / (max_dist - min_dist);
        if( score > max_score ) {
            max_score = score;
            ri = i;
        }
        
    }
    
    
    cv::Mat pulr = gray(vec[ri].rect);
    
    ptr = pulr.ptr();
    for(int r=0;r<pulr.rows;r++) {

        for(int c=0;c<pulr.cols;c++) {

            if(ptr[c] != 120 + vec[ri].tag) {
                ptr[c] = 0;
            }
        }
        ptr += pulr.step[0];
    }

    
    cv::Mat fc;
    pulr.copyTo(fc);
    cv::vector< cv::vector<cv::Point> > contours;
    cv::findContours(fc, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    
    cv::Point center = getMaxCircleCenter(pulr, contours);

    
    pulCenter.x = vec[ri].rect.x + center.x;
    pulCenter.y = vec[ri].rect.y + center.y;
    
    pulRect = vec[ri].rect;
    
}

void Pupiler::procEyeCorner(cv::Mat &img) {
    
    cv::Mat grad(img.rows, img.cols, CV_8UC1);
    
    findContourRegion(img, grad, 3);
    
    cv::threshold(grad, grad, 50, 100, CV_THRESH_BINARY);
    
    cv::Rect max_rect;
    int max_area = 0;
    int max_tag = 0;
    
    uchar *ptr = grad.ptr();
    int tag = 0;
    for(int r=0;r<grad.rows;r++) {
        
        for(int c=0;c<grad.cols;c++) {
            
            if(ptr[c] == 100) {
                
               
                cv::Rect rect;
                int area = cv::floodFill(grad, cv::Point(c,r), 120+tag, &rect);
                
                
                if(area > max_area) {
                    max_area = area;
                    max_rect = rect;
                    max_tag = tag;
                }
                
                tag++;
            }
            
        }
        ptr += grad.step[0];
    }
    
    ptr = grad.ptr();
    for(int r=0;r<grad.rows;r++) {
        
        for(int c=0;c<grad.cols;c++) {
            
            if(ptr[c] != max_tag+120) {
                
                ptr[c] = 0;
            
            }
            
        }
        ptr += grad.step[0];
    }

    
    
}