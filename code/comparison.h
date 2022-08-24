#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "projection.h"

cv::Mat process_projected(const cv::Mat &projected, int blur = 10)
{
    static cv::Mat canny_img, f, pos_hat, neg_hat;
    static cv::Mat grey = cv::Mat::zeros(projected.size(), CV_32F);
    blur = blur % 2 ? blur : blur + 1;

    cv::Canny(projected, canny_img, 40, 40*3, 3);
    canny_img.convertTo(f, CV_32F);

    cv::GaussianBlur(f, pos_hat, cv::Size(blur, blur), 0);
    cv::GaussianBlur(f, neg_hat, cv::Size(2*blur-1, 2*blur-1), 0);
    pos_hat.copyTo(grey);
    grey -= neg_hat;
    //grey = pos_hat - neg_hat;

    double minval, maxval;
    cv::minMaxLoc(grey, &minval, &maxval);
    double scale_factor = 1.0 / (2*std::max(fabs(minval), fabs(maxval)));
    grey *= scale_factor;

    return grey;
}

cv::Mat process_eros(cv::Mat eros_img)
{
    static cv::Mat eros_blurred, eros_f, eros_fn;
    //cv::GaussianBlur(eros_img, eros_blurred, cv::Size(7, 7), 0);
    eros_img.convertTo(eros_f, CV_32F, 0.003921569);
    //cv::normalize(eros_f, eros_fn, 0.0, 1.0, cv::NORM_MINMAX);

    return eros_f;

}

double similarity_score(const cv::Mat& observation, const cv::Mat& expectation)
{
    static cv::Mat muld;
    muld = expectation.mul(observation);
    return cv::sum(cv::sum(muld))[0];
}



// from the interaction matrix and current depth calculate the desired state
// change to move the image by approximately 1 pixel.

//[du dv] =
//[ fx/d, 0, -(u-cx)/d, -(u-cx)(v-cy)/fy, fx*fx+(u-cx)(u-cx)/fx, -(v-cy)fx/fy
//  0, fy/d, -(v-cy)/d, -(fy*fy)-(v-cy)(v-cy)/fy, (u-cx)(v-cy)/fx, (u-cx)fy/fx] *

// [dx, dy, dz, dalpha, dbeta, dgamma]

class predictions {

private:

    double dp{2};
    double blur{10};
    std::array<double, 6> cp;
    enum cam_param_name{w,h,cx,cy,fx,fy};
    
    std::array<double, 7> state_current;
    double d;
    std::array<double, 7> state_projection;

    cv::Size proc_size{cv::Size(100, 100)};
    cv::Rect img_roi; //roi on the image
    cv::Mat proc_proj; // proc_sz
    cv::Rect proc_roi;
    double scale{1.0};
    cv::Mat proc_obs;
    
    double score_projection;

    std::array<cv::Mat, 6> warps_p;
    std::array<cv::Mat, 6> warps_n;
    std::array<std::array<double, 7>, 6> states_p;
    std::array<std::array<double, 7>, 6> states_n;
    std::array<double, 6> scores_p = {-DBL_MAX};
    std::array<double, 6> scores_n = {-DBL_MAX};
    std::array<cv::Mat, 6> M_p;
    std::array<cv::Mat, 6> M_n;

public:

    enum axis_name{x,y,z,a,b,c};

    void initialise(const std::array<double, 6> intrinsics, cv::Size size_to_process, int blur)
    {
        cp = intrinsics;
        proc_size = size_to_process;
        this->blur = blur;

        for(int i = 0; i < warps_p.size(); i++) {
            warps_p[i] = cv::Mat::zeros(proc_size.height, proc_size.width, CV_32F);
            warps_n[i] = cv::Mat::zeros(proc_size.height, proc_size.width, CV_32F);
        }
        proc_proj = cv::Mat::zeros(proc_size.height, proc_size.width, CV_32F);
        img_roi = cv::Rect(0, 0, cp[w], cp[h]); //this gets updated with every projection
        proc_roi = cv::Rect(0, 0, proc_size.width, proc_size.height); //this gets updated with every projection
    }

    void extract_rois(const cv::Mat &projected)
    {
        int buffer = blur;
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
            scale = (double)img_roi.width / proc_roi.width;
            double ratio = (double)img_roi.height / img_roi.width;
            proc_roi.height = proc_size.height * ratio;
            proc_roi.y = (proc_size.height - proc_roi.height) * 0.5;
        } else {
            proc_roi.height = proc_size.height;
            proc_roi.y = 0;
            scale = (double)img_roi.height / proc_roi.height;
            double ratio = (double)img_roi.width / img_roi.height;
            proc_roi.width = proc_size.width * ratio;
            proc_roi.x = (proc_size.width - proc_roi.width) * 0.5;
        }
    }


    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
        //d = state[2];
        d = sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    }

    void set_projection(const std::array<double, 7> &state, const cv::Mat &image)
    {
        state_projection = state;
        static cv::Mat roi_rgb = cv::Mat::zeros(proc_size, CV_8UC3);
        roi_rgb = 0;
        //resize could use nearest to speed up?
        cv::resize(image(img_roi), roi_rgb(proc_roi), proc_roi.size());
        proc_proj = process_projected(roi_rgb, blur);
    }

    void set_observation(const cv::Mat &image)
    {
        //image comes in as a 8U and must be converted to 32F
        static cv::Mat roi_u = cv::Mat::zeros(proc_size, CV_8U);
        roi_u = 0;
        //resize could use nearest to speed up?
        cv::resize(image(img_roi), roi_u(proc_roi), proc_roi.size());
        proc_obs = process_eros(roi_u);

        for (auto &s : states_p)
            s = state_current;

        for (auto &s : states_n)
            s = state_current;

        score_projection = similarity_score(proc_obs, proc_proj);
    }

    void create_Ms(int dp)
    {
        this->dp = dp;
        static std::array<cv::Point2f, 3> dst_n, dst_p;
        static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
        src[1].x = proc_size.width; src[1].y = proc_size.height*0.25;
        src[2].x = proc_size.width*0.25; src[2].y = proc_size.height;

        cv::Point cen = cv::Point(proc_size.width * 0.5, proc_size.height*0.5);

        //x is easy
        M_p[x] = (cv::Mat_<double>(2, 3) << 1, 0, dp, 0, 1, 0);
        M_n[x] = (cv::Mat_<double>(2, 3) << 1, 0, -dp, 0, 1, 1);

        //y is easy
        M_p[y] = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, dp);
        M_n[y] = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, -dp);

        //z we use the 3 point formula
        for(int i = 0; i < dst_n.size(); i++) 
        {
            double du = -(src[i].x-cen.x) * dp * 2 / cen.x;
            double dv = -(src[i].y-cen.y) * dp * 2 / cen.y;
            dst_n[i] = cv::Point2f(du, dv);
            dst_p[i] = src[i] - dst_n[i];
            dst_n[i] = src[i] + dst_n[i];
        }
        M_p[z] = cv::getAffineTransform(src, dst_p);
        M_n[z] = cv::getAffineTransform(src, dst_n);

        //roll we use the 3 point formula
        double theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5); 
        for(int i = 0; i < dst_n.size(); i++) 
        {
            dst_n[i] = cv::Point2f(-(src[i].y - cen.y) * cp[fx] / cp[fy] * theta,
                                   (src[i].x - cen.x) * cp[fy] / cp[fx] * theta);
            dst_p[i] = src[i] - dst_n[i];
            dst_n[i] = src[i] + dst_n[i];
        }
        M_p[c] = cv::getAffineTransform(src, dst_p);
        M_n[c] = cv::getAffineTransform(src, dst_n);
    }

    void compare_to_warp_x() 
    {
        cv::warpAffine(proc_proj, warps_p[x], M_p[x], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        cv::warpAffine(proc_proj, warps_n[x], M_n[x], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        //yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[x][x] += (scale * dp * d / cp[fx]);
        states_n[x][x] -= (scale * dp * d / cp[fx]);

        scores_p[x] = similarity_score(proc_obs, warps_p[x]);
        scores_n[x] = similarity_score(proc_obs, warps_n[x]);
    }

    void compare_to_warp_y() 
    {
        cv::warpAffine(proc_proj, warps_p[y], M_p[y], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::warpAffine(proc_proj, warps_n[y], M_n[y], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        // yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[y][y] -= (scale * dp * d / cp[fy]);
        states_n[y][y] += (scale * dp * d / cp[fy]);

        // state[0] += 1 * d / cp[fx];
        scores_p[y] = similarity_score(proc_obs, warps_p[y]);
        scores_n[y] = similarity_score(proc_obs, warps_n[y]);
    }

    void compare_to_warp_z() 
    {
        cv::warpAffine(proc_proj, warps_p[z], M_p[z], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        //image_projection(roi).copyTo(warps_p[b](roi));

        cv::warpAffine(proc_proj, warps_n[z], M_n[z], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        //image_projection(roi).copyTo(warps_n[b](roi));

        // calculate the state change given interactive matrix
        states_p[z][z] += scale * dp * 2 * d / (proc_size.width*0.5);
        states_n[z][z] -= scale * dp * 2 * d / (proc_size.width*0.5);
        //perform_rotation(states_p[z], 2, -theta);
        //perform_rotation(states_n[a], 2, theta);
        
        scores_p[z] = similarity_score(proc_obs, warps_p[z]);
        scores_n[z] = similarity_score(proc_obs, warps_n[z]);
    }

    void compare_to_warp_c() 
    {
        //roll
        double theta = atan2(dp, std::max(proc_size.width, proc_size.height)*0.5);

        //three point formula//three point formula
        //du = -(v-cy)fx/fy * dc
        //dv = (u-cx)fy/fx * dc
        cv::warpAffine(proc_proj, warps_p[c], M_p[c], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        cv::warpAffine(proc_proj, warps_n[c], M_n[c], proc_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        perform_rotation(states_p[c], 0, theta);
        perform_rotation(states_n[c], 0, -theta);

        scores_p[c] = similarity_score(proc_obs, warps_p[c]);
        scores_n[c] = similarity_score(proc_obs, warps_n[c]);
    }

    // void compare_to_warp_b(const cv::Mat &obs, int dp) 
    // {
    //     //yaw
    //     //du = ((fx*fx)+(u-cx)(u-cx))/fx * db
    //     //dv = (u-cx)(v-cy)/fx * db

    //     //db = (fx*fx+(u-cx)(u-cx)/fx) / du;
    //     double theta = (cp[fx] * dp) / ((cp[fx]*cp[fx]) + (roi.width*roi.width*0.25));

    //     cv::Point cen(roi.width*0.5, roi.height*0.5);
    //     static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
    //     src[1].x = roi.width; src[1].y = roi.height*0.25;
    //     src[2].x = roi.width*0.25; src[2].y = roi.height;

    //     //three point formula//three point formula
    //     static std::array<cv::Point2f, 3> dst_n, dst_p;
    //     for(int i = 0; i < dst_n.size(); i++) 
    //     {
    //         double du = (cp[fx]*cp[fx] + (src[i].x-cen.x)*(src[i].x-cen.x))/cp[fx];
    //         double dv = ((src[i].x-cen.x)*(src[i].y-cen.y))/cp[fx];
    //         dst_n[i] = cv::Point2f(du * theta, dv * theta);
    //         dst_p[i] = src[i] - dst_n[i];
    //         dst_n[i] = src[i] + dst_n[i];
    //     }
    //     static cv::Mat M;

    //     M = cv::getAffineTransform(src, dst_p);
    //     cv::warpAffine(image_projection(roi), warps_p[b](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    //     //image_projection(roi).copyTo(warps_p[b](roi));

    //     M = cv::getAffineTransform(src, dst_n);
    //     cv::warpAffine(image_projection(roi), warps_n[b](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    //     //image_projection(roi).copyTo(warps_n[b](roi));

    //     // calculate the state change given interactive matrix
    //     perform_rotation(states_p[b], 1, -theta);
    //     perform_rotation(states_n[b], 1, theta);
        
    //     scores_p[b] = similarity_score(obs, warps_p[b](roi));
    //     scores_n[b] = similarity_score(obs, warps_n[b](roi));
    // }

    // void compare_to_warp_a(const cv::Mat &obs, int dp) 
    // {
    //     //pitch
    // //[du dv] =
    // //[ -(u-cx)(v-cy)/fy, 
    // //  -(fy*fy)-(v-cy)(v-cy)/fy] * da

    //     double theta = (cp[fy] * dp) / (-(cp[fy]*cp[fy]) - (roi.width*roi.width*0.25));

    //     cv::Point cen(roi.width*0.5, roi.height*0.5);
    //     static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
    //     src[1].x = roi.width; src[1].y = roi.height*0.25;
    //     src[2].x = roi.width*0.25; src[2].y = roi.height;

    //     //three point formula//three point formula
    //     static std::array<cv::Point2f, 3> dst_n, dst_p;
    //     for(int i = 0; i < dst_n.size(); i++) 
    //     {
    //         double dv = (-cp[fy]*cp[fy] - (src[i].y-cen.y)*(src[i].y-cen.y))/cp[fy];
    //         double du = (-(src[i].x-cen.x)*(src[i].y-cen.y))/cp[fy];
    //         dst_n[i] = cv::Point2f(du * theta, dv * theta);
    //         dst_p[i] = src[i] - dst_n[i];
    //         dst_n[i] = src[i] + dst_n[i];
    //     }
    //     static cv::Mat M;

    //     M = cv::getAffineTransform(src, dst_p);
    //     cv::warpAffine(image_projection(roi), warps_p[a](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    //     //image_projection(roi).copyTo(warps_p[b](roi));

    //     M = cv::getAffineTransform(src, dst_n);
    //     cv::warpAffine(image_projection(roi), warps_n[a](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    //     //image_projection(roi).copyTo(warps_n[b](roi));

    //     // calculate the state change given interactive matrix
    //     perform_rotation(states_p[a], 2, -theta);
    //     perform_rotation(states_n[a], 2, theta);
        
    //     scores_p[a] = similarity_score(obs, warps_p[a](roi));
    //     scores_n[a] = similarity_score(obs, warps_n[a](roi));
    // }


    std::array<double, 7> next_best()
    {

        std::array<double, 7> best_state = state_projection;
        double best_score = score_projection;// > 0 ? score_projection : 0;

        for(int i = 0; i < scores_p.size(); i++) 
        {
            if(scores_p[i] > best_score) {
                best_score = scores_p[i];
                best_state = states_p[i];
            }

            if (scores_n[i] > best_score) {
                best_score = scores_n[i];
                best_state = states_n[i];
            }
        }

        return best_state;

    }

    void score_overlay(double score, cv::Mat image)
    {
        if(score > cp[h]) score = cp[h];
        if(score < 0.0) score = 0.0;
        for(int i = 0; i < cp[w]*0.05; i++)
            for(int j = 0; j < (int)score; j++)
                image.at<float>(cp[h]-j-1, i) = 1.0;

    }

    cv::Mat create_translation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cp[h]*3, cp[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cp[h], cp[w], CV_32F);

        int col = 0; int row = 0;

        if (!proc_proj.empty()) {
            col = 1; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            cv::resize(proc_proj, tile, tile.size());
            //proc_proj.copyTo(tile);
            score_overlay(score_projection, tile);
        }

        if(!warps_n[x].empty()) {
            col = 0; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[x].copyTo(tile);
            cv::resize(warps_n[x], tile, tile.size());
            score_overlay(scores_n[x], tile);
        }

        if(!warps_p[x].empty()) {
            col = 2; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[x].copyTo(tile);
            cv::resize(warps_p[x], tile, tile.size());
            score_overlay(scores_p[x], tile);
        }

        if(!warps_n[y].empty()) {
            col = 1; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[y].copyTo(tile);
            cv::resize(warps_n[y], tile, tile.size());
            score_overlay(scores_n[y], tile);
        }

        if(!warps_p[y].empty()) {
            col = 1; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[y].copyTo(tile);
            cv::resize(warps_p[y], tile, tile.size());
            score_overlay(scores_p[y], tile);
        }

        if(!warps_n[z].empty()) {
            col = 0; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[z].copyTo(tile);
            cv::resize(warps_n[z], tile, tile.size());
            score_overlay(scores_n[z], tile);
        }

        if(!warps_p[z].empty()) {
            col = 2; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[z].copyTo(tile);
            cv::resize(warps_p[z], tile, tile.size());
            score_overlay(scores_p[z], tile);
        }
        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }

    cv::Mat create_rotation_visualisation() 
    {
        static cv::Mat joined = cv::Mat::zeros(cp[h]*3, cp[w]*3, CV_32F);
        static cv::Mat joined_scaled = cv::Mat::zeros(cp[h], cp[w], CV_32F);

        int col = 0; int row = 0;

        if (!proc_proj.empty()) {
            col = 1; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //image_projection.copyTo(tile);
            cv::resize(proc_proj, tile, tile.size());
            score_overlay(score_projection, tile);
        }

        if(!warps_n[a].empty()) {
            col = 0; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[a].copyTo(tile);
            cv::resize(warps_n[a], tile, tile.size());
            score_overlay(scores_n[a], tile);
        }

        if(!warps_p[a].empty()) {
            col = 2; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[a].copyTo(tile);
            cv::resize(warps_p[a], tile, tile.size());
            score_overlay(scores_p[a], tile);
        }

        if(!warps_n[b].empty()) {
            col = 1; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[b].copyTo(tile);
            cv::resize(warps_n[b], tile, tile.size());
            score_overlay(scores_n[b], tile);
        }

        if(!warps_p[b].empty()) {
            col = 1; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[b].copyTo(tile);
            cv::resize(warps_p[b], tile, tile.size());
            score_overlay(scores_p[b], tile);
        }

        if(!warps_n[c].empty()) {
            col = 0; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_n[c].copyTo(tile);
            cv::resize(warps_n[c], tile, tile.size());
            score_overlay(scores_n[c], tile);
        }

        if(!warps_p[c].empty()) {
            col = 2; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            //warps_p[c].copyTo(tile);
            cv::resize(warps_p[c], tile, tile.size());
            score_overlay(scores_p[c], tile);
        }
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

        
        if(!proc_proj.empty()) {
            // blue = positive space
            cv::threshold(proc_proj(proc_roi), temp, 0.0, 0.5, cv::THRESH_TOZERO);
            cv::resize(temp, temp, roi.size());
            temp.convertTo(channels[0](roi), CV_8U, 1024);
            // red = negative space
            temp = proc_proj(proc_roi) * -1.0;
            cv::threshold(temp, temp, 0.0, 0.5, cv::THRESH_TOZERO);
            cv::resize(temp, temp, roi.size());
            temp.convertTo(channels[2](roi), CV_8U, 1024);
        }

        
        

        cv::merge(channels, rgb_img);

        return rgb_img;
    }
};



void predict(std::vector<double>& state, cv::Mat& projection, int axis, double delta)
{
    //get three random points in the image (given projection image size)

    //calculate the three warped positions given the interaction matrix

    //calculate the warped affine and apply the wap to the image

    //convert the state to eurler, add the delta, convert back to quaternion



}