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

    //visual parameters
    std::array<double, 6> cam;
    enum cam_param_name{w,h,cx,cy,fx,fy};
    double dp{2};
    double blur{10};

    //size/resize parameters
    cv::Size proc_size{cv::Size(100, 100)};
    cv::Rect img_roi; //roi on the image
    cv::Rect proc_roi;
    double scale{1.0};
    cv::Mat proc_obs;
    double d;

    //states
    std::array<double, 7> state_current;
    std::array<double, 7> state_projection;

    //warps
    enum warp_name{xp, yp, zp, ap, bp, cp, xn, yn, zn, an, bn, cn};
    typedef struct warp_bundle
    {
        cv::Mat M;
        cv::Mat rmx;
        cv::Mat rmy;
        cv::Mat img_warp;
        int axis{0};
        double delta{0.0};
        double score{-DBL_MAX};
    } warp_bundle;

    warp_bundle projection;
    std::array<warp_bundle, 12> warps;

    cv::Mat process_projected(const cv::Mat &projected, int blur = 10) {
        static cv::Mat canny_img, f, pos_hat, neg_hat;
        static cv::Mat grey = cv::Mat::zeros(projected.size(), CV_32F);
        blur = blur % 2 ? blur : blur + 1;

        cv::Canny(projected, canny_img, 40, 40 * 3, 3);
        canny_img.convertTo(f, CV_32F);

        cv::GaussianBlur(f, pos_hat, cv::Size(blur, blur), 0);
        cv::GaussianBlur(f, neg_hat, cv::Size(2 * blur - 1, 2 * blur - 1), 0);
        pos_hat.copyTo(grey);
        grey -= neg_hat;
        // grey = pos_hat - neg_hat;

        double minval, maxval;
        cv::minMaxLoc(grey, &minval, &maxval);
        double scale_factor = 1.0 / (2 * std::max(fabs(minval), fabs(maxval)));
        grey *= scale_factor;

        return grey;
    }

    cv::Mat process_eros(cv::Mat eros_img) {
        static cv::Mat eros_blurred, eros_f, eros_fn;
        cv::GaussianBlur(eros_img, eros_blurred, cv::Size(5, 5), 0);
        eros_blurred.convertTo(eros_f, CV_32F, 0.003921569);
        //cv::normalize(eros_f, eros_fn, 0.0, 1.0, cv::NORM_MINMAX);

        return eros_f;
    }

    double similarity_score(const cv::Mat &observation, const cv::Mat &expectation) {
        static cv::Mat muld;
        muld = expectation.mul(observation);
        return cv::sum(cv::sum(muld))[0];
    }

public:

    enum axis_name{x=0,y=1,z=2,a=3,b=4,c=5};

    void initialise(const std::array<double, 6> intrinsics, cv::Size size_to_process, int blur)
    {
        cam = intrinsics;
        proc_size = size_to_process;
        this->blur = blur;

        proc_obs = cv::Mat::zeros(proc_size, CV_32F);
        projection.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        for(auto &warp : warps) {
            warp.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        }

        img_roi = cv::Rect(0, 0, cam[w], cam[h]); //this gets updated with every projection
        proc_roi = cv::Rect(0, 0, proc_size.width, proc_size.height); //this gets updated with every projection
    }

    void create_Ms(double dp)
    {
        this->dp = dp;
        double theta = 0;
        static std::array<cv::Point2f, 3> dst_n, dst_p;
        static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
        src[1].x = proc_size.width*0.5; src[1].y = proc_size.height*0.5;
        src[2].x = proc_size.width*0.25; src[2].y = proc_size.height;

        cv::Point cen = cv::Point(proc_size.width * 0.5, proc_size.height*0.5);

        //x is a shift
        warps[xp].axis = x;
        warps[xp].M = (cv::Mat_<double>(2, 3) << 1, 0, dp, 0, 1, 0);
        warps[xn].axis = x;
        warps[xn].M = (cv::Mat_<double>(2, 3) << 1, 0, -dp, 0, 1, 0);

        //y is a shift
        warps[yp].axis = y;
        warps[yp].M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, dp);
        warps[yn].axis = y;
        warps[yn].M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -dp);

        //z we use a scaling matrix
        //this should have the centre in the image centre (not object centre)
        //but that requires recomputing M for each different position in the image
        //for computation we are making this assumption. could be improved.
        warps[zp].axis = z;
        warps[zp].delta =  dp / proc_size.width;
        warps[zp].M = cv::getRotationMatrix2D(cen, 0, 1+dp/(proc_size.width));
        warps[zn].axis = z;
        warps[zn].delta = -dp / proc_size.width;
        warps[zn].M = cv::getRotationMatrix2D(cen, 0, 1-dp/(proc_size.width));
        
        //roll we use the 3 point formula
        src[1].x = proc_size.width*0.75; src[1].y = proc_size.height*0.25;
        src[2].x = proc_size.width*0.25; src[2].y = proc_size.height;
        theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5); 
        for(int i = 0; i < dst_n.size(); i++) 
        {
            dst_n[i] = cv::Point2f(-(src[i].y - cen.y) * cam[fx] / cam[fy] * theta,
                                   (src[i].x - cen.x) * cam[fy] / cam[fx] * theta);
            dst_p[i] = src[i] - dst_n[i];
            dst_n[i] = src[i] + dst_n[i];
        }
        warps[cp].axis = c;
        warps[cp].delta = theta;
        warps[cp].M = cv::getAffineTransform(src, dst_p);
        warps[cn].axis = c;
        warps[cn].delta = -theta;
        warps[cn].M = cv::getAffineTransform(src, dst_n);

        theta = atan2(dp, proc_size.width * 0.5);
        cv::Mat bp_rmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat bp_rmy = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat bn_rmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat bn_rmy = cv::Mat::zeros(proc_size, CV_32F);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                bp_rmx.at<float>(y, x) = x + dp * cos(0.5 * M_PI * (x - cen.x) / (proc_size.width * 0.5));
                bn_rmx.at<float>(y, x) = x - dp * cos(0.5 * M_PI * (x - cen.x) / (proc_size.width * 0.5));
                bp_rmy.at<float>(y, x) = y;
                bn_rmy.at<float>(y, x) = y;
            }
        }
        cv::convertMaps(bp_rmx, bp_rmy, warps[bp].rmx, warps[bp].rmy, CV_16SC2);
        cv::convertMaps(bn_rmx, bn_rmy, warps[bn].rmx, warps[bn].rmy, CV_16SC2);
        warps[bp].axis = b;
        warps[bp].delta = -theta;
        warps[bn].delta = theta;
        warps[bn].axis = b;

        theta = atan2(dp, proc_size.height * 0.5);
        cv::Mat ap_rmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat ap_rmy = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat an_rmx = cv::Mat::zeros(proc_size, CV_32F);
        cv::Mat an_rmy = cv::Mat::zeros(proc_size, CV_32F);
        for(int x = 0; x < proc_size.width; x++) {
            for(int y = 0; y < proc_size.height; y++) {
                ap_rmx.at<float>(y, x) = x;
                an_rmx.at<float>(y, x) = x;
                ap_rmy.at<float>(y, x) = y + dp * cos(0.5 * M_PI * (y - cen.y) / (proc_size.height * 0.5));
                an_rmy.at<float>(y, x) = y - dp * cos(0.5 * M_PI * (y - cen.y) / (proc_size.height * 0.5));
            }
        }
        cv::convertMaps(ap_rmx, ap_rmy, warps[ap].rmx, warps[ap].rmy, CV_16SC2);
        cv::convertMaps(an_rmx, an_rmy, warps[an].rmx, warps[an].rmy, CV_16SC2);
        warps[ap].axis = a;
        warps[ap].delta = -theta;
        warps[an].delta = theta;
        warps[an].axis = a;
    }

    void extract_rois(const cv::Mat &projected)
    {
        int buffer = 20;
        static cv::Rect full_roi = cv::Rect(cv::Point(0, 0), projected.size());
        
        //convert to grey
        static cv::Mat grey;
        cv::cvtColor(projected, grey, cv::COLOR_BGR2GRAY);
        
        //find the bounding rectangle and add some buffer
        img_roi = cv::boundingRect(grey);
        img_roi.x -= buffer; img_roi.y-= buffer;
        img_roi.width += buffer*2; img_roi.height += buffer*2;

        //limit the roi to the image space.        
        img_roi = img_roi & full_roi;

        //find the process rois and the scale factor
        if(img_roi.width >= img_roi.height) {
            proc_roi.width = proc_size.width;
            proc_roi.x = 0;
            scale = (double)proc_roi.width / img_roi.width;
            double ratio = (double)img_roi.height / img_roi.width;
            proc_roi.height = proc_size.height * ratio;
            proc_roi.y = (proc_size.height - proc_roi.height) * 0.5;
        } else {
            proc_roi.height = proc_size.height;
            proc_roi.y = 0;
            scale = (double)proc_roi.height / img_roi.height;
            double ratio = (double)img_roi.width / img_roi.height;
            proc_roi.width = proc_size.width * ratio;
            proc_roi.x = (proc_size.width - proc_roi.width) * 0.5;
        }
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
        d = fabs(state[2]);
        //d = sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    }

    void set_projection(const std::array<double, 7> &state, const cv::Mat &image)
    {
        state_projection = state;
        static cv::Mat roi_rgb = cv::Mat::zeros(proc_size, CV_8UC3);
        roi_rgb = 0;
        //resize could use nearest to speed up?
        cv::resize(image(img_roi), roi_rgb(proc_roi), proc_roi.size(), 0, 0, cv::INTER_CUBIC);
        projection.img_warp = process_projected(roi_rgb, blur);
    }

    void set_observation(const cv::Mat &image)
    {
        //image comes in as a 8U and must be converted to 32F
        //static cv::Mat roi_32f = cv::Mat::zeros(proc_size, CV_32F);
        //roi_u = 0;
        //resize could use nearest to speed up?
        proc_obs = 0;
        cv::Mat roi_32f = process_eros(image(img_roi));
        cv::resize(roi_32f, proc_obs(proc_roi), proc_roi.size(), 0, 0, cv::INTER_CUBIC);
        //proc_obs = process_eros(roi_u);

        projection.score = similarity_score(proc_obs, projection.img_warp);
        projection.score = projection.score < 0 ? 0 : projection.score;
    }

    void compare_to_warp_x() 
    {
        cv::warpAffine(projection.img_warp, warps[xp].img_warp, warps[xp].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::warpAffine(projection.img_warp, warps[xn].img_warp, warps[xn].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        //yInfo() << (8 * d /cp[fx]) *0.001;
        warps[xp].delta =  scale * dp;
        warps[xn].delta = -scale * dp;

        warps[xp].score = similarity_score(proc_obs, warps[xp].img_warp);
        warps[xn].score = similarity_score(proc_obs, warps[xn].img_warp);
    }

    void compare_to_warp_y() 
    {
        cv::warpAffine(projection.img_warp, warps[yp].img_warp, warps[yp].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::warpAffine(projection.img_warp, warps[yn].img_warp, warps[yn].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        // yInfo() << (8 * d /cp[fx]) *0.001;
        warps[yp].delta = -scale * dp;
        warps[yn].delta =  scale * dp;

        // state[0] += 1 * d / cp[fx];
        warps[yp].score = similarity_score(proc_obs, warps[yp].img_warp);
        warps[yn].score = similarity_score(proc_obs, warps[yn].img_warp);
    }

    void compare_to_warp_z() 
    {
        cv::warpAffine(projection.img_warp, warps[zp].img_warp, warps[zp].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::warpAffine(projection.img_warp, warps[zn].img_warp, warps[zn].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        
        warps[zp].score = similarity_score(proc_obs, warps[zp].img_warp);
        warps[zn].score = similarity_score(proc_obs, warps[zn].img_warp);
    }

    void compare_to_warp_c() 
    {
        //roll
        cv::warpAffine(projection.img_warp, warps[cp].img_warp, warps[cp].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
        cv::warpAffine(projection.img_warp, warps[cn].img_warp, warps[cn].M,
            proc_size, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        warps[cp].score = similarity_score(proc_obs, warps[cp].img_warp);
        warps[cn].score = similarity_score(proc_obs, warps[cn].img_warp);
    }

    void compare_to_warp_b() 
    {
        //yaw
        cv::remap(projection.img_warp, warps[bp].img_warp, warps[bp].rmx, warps[bp].rmy, cv::INTER_CUBIC);
        cv::remap(projection.img_warp, warps[bn].img_warp, warps[bn].rmx, warps[bn].rmy, cv::INTER_CUBIC);

        warps[bp].score = similarity_score(proc_obs, warps[bp].img_warp);
        warps[bn].score = similarity_score(proc_obs, warps[bn].img_warp);
    }

    void compare_to_warp_a() 
    {
        //pitch
        cv::remap(projection.img_warp, warps[ap].img_warp, warps[ap].rmx, warps[ap].rmy, cv::INTER_CUBIC);
        cv::remap(projection.img_warp, warps[an].img_warp, warps[an].rmx, warps[an].rmy, cv::INTER_CUBIC);

        warps[ap].score = similarity_score(proc_obs, warps[ap].img_warp);
        warps[an].score = similarity_score(proc_obs, warps[an].img_warp);
    }

    void update_state(const warp_bundle &best)
    {
        double d = fabs(state_current[z]);
        switch(best.axis) {
            case(x):
                state_current[best.axis] += best.delta * d / cam[fx];
                break;
            case(y):
                state_current[best.axis] += best.delta * d / cam[fy];
                break;
            case(z):
                state_current[best.axis] += best.delta *  d;
                break;
            case(a):
                perform_rotation(state_current, 2, best.delta);
                break;
            case(b):
                perform_rotation(state_current, 1, best.delta);
                break;
            case(c):
                perform_rotation(state_current, 0, best.delta);
                break;
        }


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

    cv::Mat make_visualisation(cv::Mat full_obs) {
        cv::Mat rgb_img, temp, temp8;
        std::vector<cv::Mat> channels;
        channels.resize(3);
        channels[0] = cv::Mat::zeros(full_obs.size(), CV_8U);
        //channels[1] = cv::Mat::zeros(full_obs.size(), CV_8U);
        channels[2] = cv::Mat::zeros(full_obs.size(), CV_8U);
        // green = events
        full_obs.copyTo(channels[1]);
        cv::Rect roi = img_roi;

        // blue = positive space
        cv::threshold(projection.img_warp(proc_roi), temp, 0.0, 0.5, cv::THRESH_TOZERO);
        cv::resize(temp, temp, roi.size());
        temp.convertTo(channels[0](roi), CV_8U, 1024);
        // red = negative space
        temp = projection.img_warp(proc_roi) * -1.0;
        cv::threshold(temp, temp, 0.0, 0.5, cv::THRESH_TOZERO);
        cv::resize(temp, temp, roi.size());
        temp.convertTo(channels[2](roi), CV_8U, 1024);

        cv::merge(channels, rgb_img);

        return rgb_img;
    }
};
