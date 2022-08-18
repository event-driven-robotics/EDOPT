#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "projection.h"

cv::Mat process_projected(const cv::Mat &projected, cv::Size psize, int blur = 10)
{
    static cv::Mat rszd_img, canny_img, f, pos_hat, neg_hat;
    static cv::Mat grey = cv::Mat::zeros(psize, CV_32F);
    blur = blur % 2 ? blur : blur + 1;

    //cv::resize(projected, rszd_img, psize);
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

cv::Mat process_eros(cv::Mat eros_img, cv::Size psize)
{
    static cv::Mat eros_blurred, eros_f, eros_fn, eros_rs;
    //cv::GaussianBlur(eros_img, eros_blurred, cv::Size(7, 7), 0);
    //cv::resize(eros_img, eros_rs, psize);
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

cv::Mat make_visualisation(cv::Mat observation, cv::Mat expectation, cv::Rect roi)
{
    cv::Mat rgb_img, temp, temp8;
    std::vector<cv::Mat> channels;
    channels.resize(3);
    channels[0] = cv::Mat::zeros(observation.size(), CV_8U);
    channels[2] = cv::Mat::zeros(observation.size(), CV_8U);
    //green = events
    observation.copyTo(channels[1]);

    // blue = positive space
    cv::threshold(expectation, temp, 0.0, 0.5, cv::THRESH_TOZERO);
    cv::resize(temp, temp, roi.size());
    temp.convertTo(channels[0](roi), CV_8U, 1024);

    // red = negative space
    temp = expectation * -1.0;
    cv::threshold(temp, temp, 0.0, 0.5, cv::THRESH_TOZERO);
    cv::resize(temp, temp, roi.size());
    temp.convertTo(channels[2](roi), CV_8U, 1024);

    cv::merge(channels, rgb_img);

    return rgb_img;
}

// from the interaction matrix and current depth calculate the desired state
// change to move the image by approximately 1 pixel.

//[du dv] =
//[ fx/d, 0, -(u-cx)/d, -(u-cx)(v-cy)/fy, fx*fx+(u-cx)(u-cx)/fx, -(v-cy)fx/fy
//  0, fy/d, -(v-cy)/d, -(fy*fy)-(v-cy)(v-cy)/fy, (u-cx)(v-cy)/fx, (u-cx)fy/fx] *

// [dx, dy, dz, dalpha, dbeta, dgamma]

class predictions {

private:
    
    enum cam_param_name{w,h,cx,cy,fx,fy};
    std::array<double, 6> cp;

    std::array<double, 7> state_current;
    double d;

    std::array<double, 7> state_projection;
    cv::Mat image_projection;
    double score_projection;
    cv::Rect roi;

    double default_dp = 8;

    std::array<cv::Mat, 6> warps_p;
    std::array<cv::Mat, 6> warps_n; 
    std::array<std::array<double, 7>, 6> states_p;
    std::array<std::array<double, 7>, 6> states_n;
    std::array<double, 6> scores_p = {-DBL_MAX};
    std::array<double, 6> scores_n = {-DBL_MAX};

public:

    enum axis_name{x,y,z,a,b,c};

    void set_intrinsics(const std::array<double, 6> parameters)
    {
        cp = parameters;
        for(int i = 0; i < warps_p.size(); i++) {
            warps_p[i] = cv::Mat::zeros(cp[h], cp[w], CV_32F);
            warps_n[i] = cv::Mat::zeros(cp[h], cp[w], CV_32F);
        }
        image_projection = cv::Mat::zeros(cp[h], cp[w], CV_32F);
        roi = cv::Rect(0, 0, cp[w], cp[h]);
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
        d = sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    }

    void set_projection(const std::array<double, 7> &state, const cv::Mat &image, const cv::Rect& next_roi)
    {
        state_projection = state;
        image_projection(roi) = 0; //clear the old projection
        image.copyTo(image_projection(next_roi)); //copy the next projection
        roi = next_roi; //set the roi
    }

    void reset_comparison(const cv::Mat &obs)
    {
        for (auto &s : states_p)
            s = state_current;

        for (auto &s : states_n)
            s = state_current;

        score_projection = similarity_score(obs, image_projection(roi));

    }

    void compare_to_warp_x(const cv::Mat &obs, int dp) 
    {
        // we want to move the image by 1 in the x axis
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 0);

        M.at<double>(0, 2) = dp;
        cv::warpAffine(image_projection(roi), warps_p[x](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        // cv::Mat proi = image_projection(roi);
        // cv::Rect left = cv::Rect(0, 0, obs.cols-dp, obs.rows);
        // cv::Rect right = cv::Rect(dp, 0, obs.cols-dp, obs.rows);

        // cv::Mat wroi_p = warps_p[x](roi);
        // wroi_p = 0;
        // wroi_p(right) = proi(left);

        M.at<double>(0, 2) = -dp;
        cv::warpAffine(image_projection(roi), warps_n[x](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // cv::Mat wroi_n = warps_n[x](roi);
        // wroi_n = 0;
        // wroi_n(left) = proi(right);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        //yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[x][x] += (dp * d /cp[fx]);
        states_n[x][x] -= (dp * d /cp[fx]);

        scores_p[x] = similarity_score(obs, warps_p[x](roi));
        scores_n[x] = similarity_score(obs, warps_n[x](roi));
    }

    void compare_to_warp_y(const cv::Mat &obs, int dp) 
    {
        // we want to move the image by 1 in the y axis
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 1);

        M.at<double>(1, 2) = dp;
        cv::warpAffine(image_projection(roi), warps_p[y](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M.at<double>(1, 2) = -dp;
        cv::warpAffine(image_projection(roi), warps_n[y](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        // yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[y][y] -= (dp * d / cp[fy]);
        states_n[y][y] += (dp * d / cp[fy]);

        // state[0] += 1 * d / cp[fx];
        scores_p[y] = similarity_score(obs, warps_p[y](roi));
        scores_n[y] = similarity_score(obs, warps_n[y](roi));
    }

    void compare_to_warp_z(const cv::Mat &obs, int dp)  
    {
        //how to get the distance? DM = max(object distance from centre)
        // lets move m pixel -> D% = 1 - m / DM
        // lets move m pixel -> d% = 1 + m / DM
        static cv::Mat M;
        //cv::Rect roi = cv::boundingRect(image_projection);
        double dmx = std::max(fabs(roi.x - cp[cx]), fabs(roi.x + roi.width - cp[cx]));
        double dmy = std::max(fabs(roi.y - cp[cy]), fabs(roi.y + roi.height - cp[cy]));
        //  double dm = sqrt(dmx*dmx + dmy*dmy);
        
        //double dm = (cp[w] * 0.5);
        double dm = std::max(dmx, dmy);
        //double dperc = dp / dm; 

        //yInfo() << dmx << 1-dperc << 1+dperc;
        cv::Rect roi_small = cv::Rect(roi.x+dp, roi.y+dp, roi.width-2*dp, roi.height-2*dp);
        cv::Rect roi_big = cv::Rect(roi.x-dp, roi.y-dp, roi.width+2*dp, roi.height+2*dp);

        cv::resize(image_projection(roi), warps_p[z](roi_big), roi_big.size(), 0, 0, cv::INTER_NEAREST);
        cv::resize(image_projection(roi), warps_n[z](roi_small), roi_small.size(), 0, 0, cv::INTER_NEAREST);
        
        // M = cv::getRotationMatrix2D(cv::Point(cp[cx], cp[cy]), 0, 1 - dperc);
        // cv::warpAffine(image_projection, warps_n[z], M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // M = cv::getRotationMatrix2D(cv::Point(cp[cx], cp[cy]), 0, 1 + dperc);
        // cv::warpAffine(image_projection, warps_p[z], M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        //how to update the state
        //du = -(u-cx)/d * dz -> dz = du * d / -(c-cx) -> dz = dpix * d / dm
        //dv = -(v-cy)/d * dz
        states_p[z][z] += dp * d / dm;
        states_n[z][z] -= dp * d / dm;

        scores_p[z] = similarity_score(obs, warps_p[z](roi));
        scores_n[z] = similarity_score(obs, warps_n[z](roi));
    }

    void compare_to_warp_c(const cv::Mat &obs, int dp) 
    {
        //roll

        //angle to rotate by 
        double theta = atan2(dp, std::max(roi.width, roi.height)*0.5);

        //three point formula//three point formula
        //du = -(v-cy)fx/fy * dc
        //dv = (u-cx)fy/fx * dc
        static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
        src[1].x = roi.width;
        src[2].y = roi.height;

        cv::Point cen(roi.width*0.5, roi.height*0.5);

        static std::array<cv::Point2f, 3> dst_n, dst_p;
        for(int i = 0; i < dst_n.size(); i++) 
        {
            dst_n[i] = cv::Point2f(-(src[i].y - cen.y) * cp[fx] / cp[fy] * theta,
                                   (src[i].x - cen.x) * cp[fy] / cp[fx] * theta);
            dst_p[i] = src[i] - dst_n[i];
            dst_n[i] = src[i] + dst_n[i];
        }
        static cv::Mat M;

        M = cv::getAffineTransform(src, dst_p);
        cv::warpAffine(image_projection(roi), warps_p[c](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M = cv::getAffineTransform(src, dst_n);
        cv::warpAffine(image_projection(roi), warps_n[c](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        perform_rotation(states_p[c], 0, theta);
        perform_rotation(states_n[c], 0, -theta);

        scores_p[c] = similarity_score(obs, warps_p[c](roi));
        scores_n[c] = similarity_score(obs, warps_n[c](roi));
    }

    void compare_to_warp_b(const cv::Mat &obs, int dp) 
    {
        //yaw

        //angle to rotate by 
        double theta = atan2(dp, std::max(roi.width, roi.height)*0.5);

        //three point formula//three point formula
        //du = -(v-cy)fx/fy * dc
        //dv = (u-cx)fy/fx * dc
        static std::array<cv::Point2f, 3> src{cv::Point(0, 0)};
        src[1].x = roi.width;
        src[2].y = roi.height;

        cv::Point cen(roi.width*0.5, roi.height*0.5);

        static std::array<cv::Point2f, 3> dst_n, dst_p;
        for(int i = 0; i < dst_n.size(); i++) 
        {
            dst_n[i] = cv::Point2f(-(src[i].y - cen.y) * cp[fx] / cp[fy] * theta,
                                   (src[i].x - cen.x) * cp[fy] / cp[fx] * theta);
            dst_p[i] = src[i] - dst_n[i];
            dst_n[i] = src[i] + dst_n[i];
        }
        static cv::Mat M;

        M = cv::getAffineTransform(src, dst_p);
        cv::warpAffine(image_projection(roi), warps_p[c](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M = cv::getAffineTransform(src, dst_n);
        cv::warpAffine(image_projection(roi), warps_n[c](roi), M, roi.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        perform_rotation(states_p[c], 0, theta);
        perform_rotation(states_n[c], 0, -theta);
        
        scores_p[c] = similarity_score(obs, warps_p[c](roi));
        scores_n[c] = similarity_score(obs, warps_n[c](roi));
    }


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

        if (!image_projection.empty()) {
            col = 1; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            image_projection.copyTo(tile);
            score_overlay(score_projection, tile);
        }

        if(!warps_n[x].empty()) {
            col = 0; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[x].copyTo(tile);
            score_overlay(scores_n[x], tile);
        }

        if(!warps_p[x].empty()) {
            col = 2; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[x].copyTo(tile);
            score_overlay(scores_p[x], tile);
        }

        if(!warps_n[y].empty()) {
            col = 1; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[y].copyTo(tile);
            score_overlay(scores_n[y], tile);
        }

        if(!warps_p[y].empty()) {
            col = 1; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[y].copyTo(tile);
            score_overlay(scores_p[y], tile);
        }

        if(!warps_n[z].empty()) {
            col = 0; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[z].copyTo(tile);
            score_overlay(scores_n[z], tile);
        }

        if(!warps_p[z].empty()) {
            col = 2; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[z].copyTo(tile);
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

        if (!image_projection.empty()) {
            col = 1; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            image_projection.copyTo(tile);
            score_overlay(score_projection, tile);
        }

        if(!warps_n[a].empty()) {
            col = 0; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[a].copyTo(tile);
            score_overlay(scores_n[a], tile);
        }

        if(!warps_p[a].empty()) {
            col = 2; row = 1;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[a].copyTo(tile);
            score_overlay(scores_p[a], tile);
        }

        if(!warps_n[b].empty()) {
            col = 1; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[b].copyTo(tile);
            score_overlay(scores_n[b], tile);
        }

        if(!warps_p[b].empty()) {
            col = 1; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[b].copyTo(tile);
            score_overlay(scores_p[b], tile);
        }

        if(!warps_n[c].empty()) {
            col = 0; row = 0;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[c].copyTo(tile);
            score_overlay(scores_n[c], tile);
        }

        if(!warps_p[c].empty()) {
            col = 2; row = 2;
            cv::Mat tile = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[c].copyTo(tile);
            score_overlay(scores_p[c], tile);
        }
        cv::resize(joined, joined_scaled, joined_scaled.size());

        return joined_scaled;

    }
};



void predict(std::vector<double>& state, cv::Mat& projection, int axis, double delta)
{
    //get three random points in the image (given projection image size)

    //calculate the three warped positions given the interaction matrix

    //calculate the warped affine and apply the wap to the image

    //convert the state to eurler, add the delta, convert back to quaternion



}