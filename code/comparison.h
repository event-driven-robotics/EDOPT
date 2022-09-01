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
        cv::Mat rmp;
        cv::Mat rmsp;
        cv::Mat img_warp;
        int axis{0};
        double delta{0.0};
        double score{-DBL_MAX};
        bool active{false};
        warp_name name;
    } warp_bundle;

    warp_bundle projection;
    std::array<warp_bundle, 12> warps;
    std::deque<warp_name> warp_history;

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
        projection.axis = -1;
        projection.delta = 0;
        for(auto &warp : warps) {
            warp.img_warp = cv::Mat::zeros(proc_size, CV_32F);
        }

        img_roi = cv::Rect(0, 0, cam[w], cam[h]); //this gets updated with every projection
        proc_roi = cv::Rect(0, 0, proc_size.width, proc_size.height); //this gets updated with every projection

        warps[xp].active = warps[xn].active = true; 
        warps[yp].active = warps[yn].active = true; 
        warps[zp].active = warps[zn].active = true; 
        warps[ap].active = warps[an].active = true; 
        warps[bp].active = warps[bn].active = true; 
        warps[cp].active = warps[cn].active = true;
        warps[xp].name = xp; warps[xn].name = xn;
        warps[yp].name = yp; warps[yn].name = yn;
        warps[zp].name = zp; warps[zn].name = zn;
        warps[ap].name = ap; warps[an].name = an;
        warps[bp].name = bp; warps[bn].name = bn;
        warps[cp].name = cp; warps[cn].name = cn;
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

        static std::array<cv::Point2f, 3> dst_n, dst_p;
        static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
        src[1].x = proc_size.width*0.5; src[1].y = proc_size.height*0.5;
        src[2].x = proc_size.width*0.25; src[2].y = proc_size.height;

        cv::Point cen = cv::Point(proc_size.width * 0.5, proc_size.height*0.5);

        //z we use a scaling matrix
        //this should have the centre in the image centre (not object centre)
        //but that requires recomputing M for each different position in the image
        //for computation we are making this assumption. could be improved.
        warps[zp].axis = z;
        warps[zp].delta = dp / proc_size.width;
        warps[zp].M = cv::getRotationMatrix2D(cen, 0, 1-dp/(proc_size.width));
        warps[zn].axis = z;
        warps[zn].delta = -dp / proc_size.width;
        warps[zn].M = cv::getRotationMatrix2D(cen, 0, 1+dp/(proc_size.width));
        
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
        //projection.score = projection.score < 0 ? 0 : projection.score;
    }

    void warp()
    {
        // X
        if (warps[xp].active) {
            cv::remap(projection.img_warp, warps[xp].img_warp, warps[xp].rmp, warps[xp].rmsp,cv::INTER_LINEAR);
        }
        if (warps[xn].active) {
            cv::remap(projection.img_warp, warps[xn].img_warp, warps[xn].rmp, warps[xn].rmsp, cv::INTER_LINEAR);
        }

        // Y
        if (warps[yp].active) {
            cv::remap(projection.img_warp, warps[yp].img_warp, warps[yp].rmp, warps[yp].rmsp,cv::INTER_LINEAR);
        }
        if (warps[yn].active) {
            cv::remap(projection.img_warp, warps[yn].img_warp, warps[yn].rmp, warps[yn].rmsp,cv::INTER_LINEAR);
        }

        // Z
        if (warps[zp].active) {
            cv::warpAffine(projection.img_warp, warps[zp].img_warp, warps[zp].M,
                           proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        }
        if (warps[zn].active) {
            cv::warpAffine(projection.img_warp, warps[zn].img_warp, warps[zn].M,
                           proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        }

        // A
        if (warps[ap].active) {
            cv::remap(projection.img_warp, warps[ap].img_warp, warps[ap].rmp, warps[ap].rmsp, cv::INTER_LINEAR);
        }
        if (warps[an].active) {
            cv::remap(projection.img_warp, warps[an].img_warp, warps[an].rmp, warps[an].rmsp, cv::INTER_LINEAR);
        }

        // B
        if (warps[bp].active) {
            cv::remap(projection.img_warp, warps[bp].img_warp, warps[bp].rmp, warps[bp].rmsp, cv::INTER_LINEAR);
        }
        if (warps[bn].active) {
            cv::remap(projection.img_warp, warps[bn].img_warp, warps[bn].rmp, warps[bn].rmsp, cv::INTER_LINEAR);
        }

        // C
        if (warps[cp].active) {
            cv::remap(projection.img_warp, warps[cp].img_warp, warps[cp].rmp, warps[cp].rmsp, cv::INTER_LINEAR);
        }
        if (warps[cn].active) {
            cv::remap(projection.img_warp, warps[cn].img_warp, warps[cn].rmp, warps[cn].rmsp, cv::INTER_LINEAR);
        }
    }

    void score()
    {
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
        //warp_history.push_back(best.name);
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
