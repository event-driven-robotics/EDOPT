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
    enum warp_name{xp, yp, zp, ap, bp, cp, xn, yn, zn, an, bn, cn,
                   xp2, yp2, zp2, ap2, bp2, cp2, xn2, yn2, zn2, an2, bn2, cn2};
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
    cv::Size proc_size{cv::Size(100, 100)};

    //parameters that must be udpated externally
    double scale{1.0};
    cv::Mat proc_obs;

    //internal variables
    warp_bundle projection;
    std::array<warp_bundle, 24> warps;
    std::deque<const warp_bundle *> warp_history;
    cv::Mat prmx;
    cv::Mat prmy;
    cv::Mat nrmx;
    cv::Mat nrmy;

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

        prmx = cv::Mat::zeros(proc_size, CV_32F);
        prmy = cv::Mat::zeros(proc_size, CV_32F);
        nrmx = cv::Mat::zeros(proc_size, CV_32F);
        nrmy = cv::Mat::zeros(proc_size, CV_32F);

        warps[xp].active = warps[xn].active = true; 
        warps[yp].active = warps[yn].active = true; 
        warps[zp].active = warps[zn].active = true; 
        warps[ap].active = warps[an].active = true; 
        warps[bp].active = warps[bn].active = true; 
        warps[cp].active = warps[cn].active = true;

        warps[xp2].active = warps[xn2].active = true; 
        warps[yp2].active = warps[yn2].active = true; 
        warps[zp2].active = warps[zn2].active = true; 
        warps[ap2].active = warps[an2].active = true; 
        warps[bp2].active = warps[bn2].active = true; 
        warps[cp2].active = warps[cn2].active = true;
    }

    void create_m_x(double dp, warp_name p, warp_name n)
    {
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = x; warps[n].axis = x;
        warps[p].delta = dp; warps[n].delta = -dp;
    }

    void create_m_y(double dp, warp_name p, warp_name n)
    {
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = y; warps[n].axis = y;
        warps[p].delta = dp; warps[n].delta = -dp;
    }

    void create_m_z(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = z; warps[n].axis = z;
        warps[p].delta = dp / proc_size.width; 
        warps[n].delta = -dp / proc_size.width;
    }

    void create_m_a(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        double theta = atan2(dp, proc_size.height * 0.5);
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = a; warps[n].axis = a;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_m_b(double dp, warp_name p, warp_name n)
    {
        
        double theta = atan2(dp, proc_size.width * 0.5);
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = b; warps[n].axis = b;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_m_c(double dp, warp_name p, warp_name n)
    {
        double cy = proc_size.height * 0.5;
        double cx = proc_size.width  * 0.5;
        double theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5);
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
        cv::convertMaps(prmx, prmy, warps[p].rmp, warps[p].rmsp, CV_16SC2);
        cv::convertMaps(nrmx, nrmy, warps[n].rmp, warps[n].rmsp, CV_16SC2);
        warps[p].axis = c; warps[n].axis = c;
        warps[p].delta = theta; warps[n].delta = -theta;
    }

    void create_Ms(double dp)
    {
        create_m_x(dp, xp, xn);
        create_m_y(dp, yp, yn);
        create_m_z(dp, zp, zn);
        create_m_a(dp, ap, an);
        create_m_b(dp, bp, bn);
        create_m_c(dp, cp, cn);

        create_m_x(dp*2, xp2, xn2);
        create_m_y(dp*2, yp2, yn2);
        create_m_z(dp*2, zp2, zn2);
        create_m_a(dp*2, ap2, an2);
        create_m_b(dp*2, bp2, bn2);
        create_m_c(dp*2, cp2, cn2);
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
    }

    void make_predictive_warps()
    {
        for(auto &warp : warps)
            if(warp.active)
                cv::remap(projection.img_warp, warp.img_warp, warp.rmp, warp.rmsp, cv::INTER_LINEAR);
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
        projection.score = projection.score < 0 ? 0 : projection.score;
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
        cv::remap(projection.img_warp, projection.img_warp, best.rmp, best.rmsp, cv::INTER_LINEAR);
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

    bool update_heuristically()
    {
        int prev_history_length = warp_history.size();
        //best of x axis and roation around y (yaw)
        warp_bundle *best;
        best = &projection;
        if (warps[xp].score > best->score) best = &warps[xp];
        if (warps[xn].score > best->score) best = &warps[xn];
        if (warps[bp].score > best->score) best = &warps[bp];
        if (warps[bn].score > best->score) best = &warps[bn];
        if (warps[xp2].score > best->score) best = &warps[xp2];
        if (warps[xn2].score > best->score) best = &warps[xn2];
        if (warps[bp2].score > best->score) best = &warps[bp2];
        if (warps[bn2].score > best->score) best = &warps[bn2];
        if(best != &projection) update_state(*best);

        //best of y axis and rotation around x (pitch)
        best = &projection;
        if (warps[yp].score > best->score) best = &warps[yp];
        if (warps[yn].score > best->score) best = &warps[yn];
        if (warps[ap].score > best->score) best = &warps[ap];
        if (warps[an].score > best->score) best = &warps[an];
        if (warps[yp2].score > best->score) best = &warps[yp2];
        if (warps[yn2].score > best->score) best = &warps[yn2];
        if (warps[ap2].score > best->score) best = &warps[ap2];
        if (warps[an2].score > best->score) best = &warps[an2];
        if(best != &projection) update_state(*best);

        //best of roll
        best = &projection;
        if (warps[cp].score > best->score) best = &warps[cp];
        if (warps[cn].score > best->score) best = &warps[cn];
        if (warps[cp2].score > best->score) best = &warps[cp2];
        if (warps[cn2].score > best->score) best = &warps[cn2];
        if(best != &projection) update_state(*best);

        //best of z
        best = &projection;
        if (warps[zp].score > best->score) best = &warps[zp];
        if (warps[zn].score > best->score) best = &warps[zn];
        if (warps[zp2].score > best->score) best = &warps[zp2];
        if (warps[zn2].score > best->score) best = &warps[zn2];
        if(best != &projection) update_state(*best);

        return warp_history.size() > prev_history_length;
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
            if(!warp.active) continue;
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
            if(!warp.active) continue;
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

};
