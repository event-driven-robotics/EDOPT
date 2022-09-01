#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "projection.h"

// from the interaction matrix and current depth calculate the desired state
// change to move the image by approximately 1 pixel.

//[du dv] =
//[ fx/d, 0, -(u-cx)/d, -(u-cx)(v-cy)/fy, fx*fx+(u-cx)(u-cx)/fx, -(v-cy)fx/fy
//  0, fy/d, -(v-cy)/d, -(fy*fy)-(v-cy)(v-cy)/fy, (u-cx)(v-cy)/fx, (u-cx)fy/fx] *

// [dx, dy, dz, dalpha, dbeta, dgamma]

class predictions {

public:

    //internal definitions
    enum cam_param_name{w,h,cx,cy,fx,fy};
    enum warp_name{xp, yp, zp, ap, bp, cp, xn, yn, zn, an, bn, cn};
    enum axis_name{x=0,y=1,z=2,a=3,b=4,c=5};
    typedef struct warp_bundle
    {
        cv::Mat M;
        cv::Mat rmp;
        cv::Mat rmsp;
        cv::Mat img_warp;
        int axis{0};
        double delta{0.0};
        double score{-DBL_MAX};
        bool active{false};
    } warp_bundle;

    //fixed parameters to set
    std::array<double, 6> cam;
    double dp{2};
    cv::Size proc_size{cv::Size(100, 100)};

    //parameters that must be udpated externally
    double scale{1.0};
    cv::Mat proc_obs;

    //internal variables
    warp_bundle projection;
    std::array<warp_bundle, 12> warps;
    std::deque<const warp_bundle *> warp_history;

    //output
    std::array<double, 7> state_current;

public:

    void initialise(const std::array<double, 6> intrinsics, int size_to_process)
    {
        cam = intrinsics;
        proc_size = cv::Size(size_to_process, size_to_process);

        projection.axis = -1;
        projection.delta = 0;
        proc_obs = cv::Mat::zeros(proc_size, CV_32F);
        projection.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        for(auto &warp : warps) {
            warp.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        }

        warps[xp].active = warps[xn].active = true; 
        warps[yp].active = warps[yn].active = true; 
        warps[zp].active = warps[zn].active = true; 
        warps[ap].active = warps[an].active = true; 
        warps[bp].active = warps[bn].active = true; 
        warps[cp].active = warps[cn].active = true;
    }

    void create_Ms(double dp)
    {
        //variable set-up
        this->dp = dp;
        double theta = 0;

        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;

        cv::Mat prmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat prmy = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat nrmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat nrmy = cv::Mat::zeros(proc_size, CV_32F);

        //x shift by dp
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                //positive
                prmx.at<float>(y, x) = x-dp;
                prmy.at<float>(y, x) = y;
                //negative
                nrmx.at<float>(y, x) = x+dp;
                nrmy.at<float>(y, x) = y;
            }
        }
        cv::convertMaps(prmx, prmy, warps[xp].rmp, warps[xp].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[xn].rmp, warps[xn].rmsp, CV_16SC2);
        warps[xp].axis = x; warps[xn].axis = x;
        warps[xp].delta = dp; warps[xn].delta = -dp;

        //y shift by dp
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                //positive
                prmx.at<float>(y, x) = x;
                prmy.at<float>(y, x) = y-dp;
                //negative
                nrmx.at<float>(y, x) = x;
                nrmy.at<float>(y, x) = y+dp;
            }
        }
        cv::convertMaps(prmx, prmy, warps[yp].rmp, warps[yp].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[yn].rmp, warps[yn].rmsp, CV_16SC2);
        warps[yp].axis = y; warps[yn].axis = y;
        warps[yp].delta = dp; warps[yn].delta = -dp;

        //z we use a scaling matrix
        //this should have the centre in the image centre (not object centre)
        //but that requires recomputing M for each different position in the image
        //for computation we are making this assumption. could be improved.
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -(x-cx) * dp / proc_size.width;
                double dy = -(y-cy) * dp / proc_size.height;
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[zp].rmp, warps[zp].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[zn].rmp, warps[zn].rmsp, CV_16SC2);
        warps[zp].axis = z; warps[zn].axis = z;
        warps[zp].delta = dp / proc_size.width; 
        warps[zn].delta = -dp / proc_size.width;
        
        //roll by 2 pixels at proc_width/2 distance
        theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -(y - cy) * cam[fx] / cam[fy] * theta;
                double dy =  (x - cx) * cam[fy] / cam[fx] * theta;
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[cp].rmp, warps[cp].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[cn].rmp, warps[cn].rmsp, CV_16SC2);
        warps[cp].axis = c; warps[cn].axis = c;
        warps[cp].delta = theta; warps[cn].delta = -theta;

        //yaw assuming a sphere the size of the proc_width
        theta = atan2(dp, proc_size.width * 0.5);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dx = -dp * cos(0.5 * M_PI * (x - cx) / (proc_size.width * 0.5));
                //positive
                prmx.at<float>(y, x) = x - dx;
                prmy.at<float>(y, x) = y;
                //negative
                nrmx.at<float>(y, x) = x + dx;
                nrmy.at<float>(y, x) = y;
            }
        }
        cv::convertMaps(prmx, prmy, warps[bp].rmp, warps[bp].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[bn].rmp, warps[bn].rmsp, CV_16SC2);
        warps[bp].axis = b; warps[bn].axis = b;
        warps[bp].delta = theta; warps[bn].delta = -theta;
        

        //pitch assuming the object is a sphere the size of proc_height
        theta = atan2(dp, proc_size.height * 0.5);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                double dy = dp * cos(0.5 * M_PI * (y - cy) / (proc_size.height * 0.5));
                //positive
                prmx.at<float>(y, x) = x;
                prmy.at<float>(y, x) = y - dy;
                //negative
                nrmx.at<float>(y, x) = x;
                nrmy.at<float>(y, x) = y + dy;
            }
        }
        cv::convertMaps(prmx, prmy, warps[ap].rmp, warps[ap].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[an].rmp, warps[an].rmsp, CV_16SC2);
        warps[ap].axis = a; warps[an].axis = a;
        warps[ap].delta = theta; warps[an].delta = -theta;
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
    }

    void make_predictive_warps()
    {
        for(auto &warp : warps)
            if(warp.active)
                cv::remap(projection.img_warp, warp.img_warp, warp.rmp, warp.rmsp,cv::INTER_LINEAR);
    }

    void warp_by_history(cv::Mat &image)
    {
        for(auto warp : warp_history)
            cv::remap(image, image, warp->rmp, warp->rmsp, cv::INTER_LINEAR);
        warp_history.clear();
    }

    double similarity_score(const cv::Mat &observation, const cv::Mat &expectation) {
        static cv::Mat muld;
        muld = expectation.mul(observation);
        return cv::sum(cv::sum(muld))[0];
    }

    void score_predictive_warps()
    {
        projection.score = similarity_score(proc_obs, projection.img_warp);
        for(auto &w : warps)
            if(w.active) w.score = similarity_score(proc_obs, w.img_warp);
    }

    void update_state(const warp_bundle &best)
    {
        double d = fabs(state_current[z]);
        switch(best.axis) {
            case(x):
                state_current[best.axis] += best.delta * scale * d / cam[fx];
                break;
            case(y):
                state_current[best.axis] += best.delta * scale * d / cam[fy];
                break;
            case(z):
                state_current[best.axis] += best.delta *  d;
                break;
            case(a):
                perform_rotation(state_current, 0, best.delta);
                break;
            case(b):
                perform_rotation(state_current, 1, best.delta);
                break;
            case(c):
                perform_rotation(state_current, 2, best.delta);
                break;
        }
        warp_history.push_back(&best);
    }

    void update_all_possible()
    {
        for(auto &warp : warps)
            if(warp.score > projection.score)
                update_state(warp);
    }

    void update_from_max()
    {
        warp_bundle *best = &projection;
        for(auto &warp : warps)
            if(warp.score > best->score)
                best = &warp;

        update_state(*best);
    }

    void update_heuristically()
    {
        //best of x axis and roation around y (yaw)
        warp_bundle *best;
        best = &projection;
        if (warps[xp].score > best->score)
            best = &warps[xp];
        if (warps[xn].score > best->score)
            best = &warps[xn];
        if (warps[bp].score > best->score)
            best = &warps[bp];
        if (warps[bn].score > best->score)
            best = &warps[bn];
        if(best->score > projection.score)
            update_state(*best);

        //best of y axis and rotation around x (pitch)
        best = &projection;
        if (warps[yp].score > best->score)
            best = &warps[yp];
        if (warps[yn].score > best->score)
            best = &warps[yn];
        if (warps[ap].score > best->score)
            best = &warps[ap];
        if (warps[an].score > best->score)
            best = &warps[an];
        if(best->score > projection.score)
            update_state(*best);

        //best of roll
        best = &projection;
        if (warps[cp].score > best->score)
            best = &warps[cp];
        if (warps[cn].score > best->score)
            best = &warps[cn];
        if(best->score > projection.score)
            update_state(*best);

        //best of z
        best = &projection;
        if (warps[zp].score > best->score)
            best = &warps[zp];
        if (warps[zn].score > best->score)
            best = &warps[zn];
        if(best->score > projection.score)
            update_state(*best);
    }

    void score_overlay(double score, cv::Mat image)
    {
        if(score > cam[h]) score = cam[h];
        if(score < 0.0) score = 0.0;
        for(int i = 0; i < cam[w]*0.05; i++)
            for(int j = 0; j < (int)score; j++)
                image.at<float>(cam[h]-j-1, i) = 1.0;

    }

    cv::Mat create_translation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cam[h]*3, cam[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cam[h], cam[w], CV_32F);
        cv::Mat tile;
        int col = 0; int row = 0;

        for(auto &warp : warps) {
            switch(warp.axis) {
                case(x):
                    row = 1;
                    col = warp.delta > 0 ? 2 : 0;
                    break;
                case(y):
                    col = 1;
                    row = warp.delta > 0 ? 0 : 2;
                    break;
                case(z):
                    col = warp.delta > 0 ? 2 : 0;
                    row = warp.delta > 0 ? 2 : 0;
                    break;
                default:
                    continue;
            }
            tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
            cv::resize(warp.img_warp, tile, tile.size());
            score_overlay(warp.score, tile);
        }
        col = 1;
        row = 1;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(projection.img_warp, tile, tile.size());
        score_overlay(projection.score, tile);

        col = 0;
        row = 2;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(proc_obs, tile, tile.size());

        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }

    cv::Mat create_rotation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cam[h]*3, cam[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cam[h], cam[w], CV_32F);
        cv::Mat tile;
        int col = 0; int row = 0;

        for(auto &warp : warps) {
            switch(warp.axis) {
                case(b):
                    row = 1;
                    col = warp.delta > 0 ? 2 : 0;
                    break;
                case(a):
                    col = 1;
                    row = warp.delta > 0 ? 0 : 2;
                    break;
                case(c):
                    col = warp.delta > 0 ? 2 : 0;
                    row = warp.delta > 0 ? 2 : 0;
                    break;
                default:
                    continue;
            }
            tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
            cv::resize(warp.img_warp, tile, tile.size());
            score_overlay(warp.score, tile);
        }
        col = 1;
        row = 1;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(projection.img_warp, tile, tile.size());
        score_overlay(projection.score, tile);

        col = 0;
        row = 2;
        tile = joined(cv::Rect(cam[w] * col, cam[h] * row, cam[w], cam[h]));
        cv::resize(proc_obs, tile, tile.size());

        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }

    // cv::Mat make_visualisation(cv::Mat full_obs) {
    //     cv::Mat rgb_img, temp, temp8;
    //     std::vector<cv::Mat> channels;
    //     channels.resize(3);
    //     channels[0] = cv::Mat::zeros(full_obs.size(), CV_8U);
    //     //channels[1] = cv::Mat::zeros(full_obs.size(), CV_8U);
    //     channels[2] = cv::Mat::zeros(full_obs.size(), CV_8U);
    //     // green = events
    //     full_obs.copyTo(channels[1]);
    //     cv::Rect roi = img_roi;

    //     // blue = positive space
    //     cv::threshold(projection.img_warp(proc_roi), temp, 0.0, 0.5, cv::THRESH_TOZERO);
    //     cv::resize(temp, temp, roi.size());
    //     temp.convertTo(channels[0](roi), CV_8U, 1024);
    //     // red = negative space
    //     temp = projection.img_warp(proc_roi) * -1.0;
    //     cv::threshold(temp, temp, 0.0, 0.5, cv::THRESH_TOZERO);
    //     cv::resize(temp, temp, roi.size());
    //     temp.convertTo(channels[2](roi), CV_8U, 1024);

    //     cv::merge(channels, rgb_img);

    //     return rgb_img;
    // }
};
