#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

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

double similarity_score(cv::Mat observation, cv::Mat expectation)
{
    cv::Mat muld = expectation.mul(observation);
    return cv::sum(cv::sum(muld))[0];
}

cv::Mat make_visualisation(cv::Mat observation, cv::Mat expectation)
{
    cv::Mat rgb_img, temp, temp8;
    std::vector<cv::Mat> channels;
    channels.resize(3);

    //blue = positive space
    cv::threshold(expectation, temp, 0.0, 0.5, cv::THRESH_TOZERO);
    temp.convertTo(channels[0], CV_8U, 1024);

    //green = events
    observation.convertTo(channels[1], CV_8U, 200);

    //red = negative space
    temp = expectation * -1.0;
    cv::threshold(temp, temp, 0.0, 0.5, cv::THRESH_TOZERO);
    temp.convertTo(channels[2], CV_8U, 1024);

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
    cv::Rect roi_projection;

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
    }

    void set_current(const std::array<double, 7> &state)
    {
        state_current = state;
        d = sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);
    }

    void set_projection(const std::array<double, 7> &state, const cv::Mat &image)
    {
        state_projection = state;
        image.copyTo(image_projection);
    }

    void reset_comparison(const cv::Mat &obs)
    {
        for (auto &s : states_p)
            s = state_current;

        for (auto &s : states_n)
            s = state_current;

        score_projection = similarity_score(obs, image_projection);

    }

    void extract_roi(cv::Mat projected)
    {
        static cv::Mat grey;
        cv::cvtColor(projected, grey, cv::COLOR_BGR2GRAY);
        roi_projection = cv::boundingRect(grey);

    }

    void compare_to_warp_x(const cv::Mat &obs, int dp) 
    {
        // we want to move the image by 1 in the x axis
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 0);

        M.at<double>(0, 2) = dp;
        cv::warpAffine(image_projection, warps_p[x], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M.at<double>(0, 2) = -dp;
        cv::warpAffine(image_projection, warps_n[x], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        //yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[x][x] += (dp * d /cp[fx]);
        states_n[x][x] -= (dp * d /cp[fx]);

        scores_p[x] = similarity_score(obs, warps_p[x]);
        scores_n[x] = similarity_score(obs, warps_n[x]);
    }

    void compare_to_warp_y(const cv::Mat &obs, int dp) 
    {
        // we want to move the image by 1 in the y axis
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 1);

        M.at<double>(1, 2) = dp;
        cv::warpAffine(image_projection, warps_p[y], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M.at<double>(1, 2) = -dp;
        cv::warpAffine(image_projection, warps_n[y], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        // yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[y][y] -= (dp * d / cp[fy]);
        states_n[y][y] += (dp * d / cp[fy]);

        // state[0] += 1 * d / cp[fx];
        scores_p[y] = similarity_score(obs, warps_p[y]);
        scores_n[y] = similarity_score(obs, warps_n[y]);
    }

    void compare_to_warp_z(const cv::Mat &obs, int dp)  
    {
        //how to get the distance? DM = max(object distance from centre)
        // lets move m pixel -> D% = 1 - m / DM
        // lets move m pixel -> d% = 1 + m / DM
        static cv::Mat M;
        //cv::Rect roi = cv::boundingRect(image_projection);
        double dmx = std::max(fabs(roi_projection.x - cp[cx]), fabs(roi_projection.x + roi_projection.width - cp[cx]));
        double dmy = std::max(fabs(roi_projection.y - cp[cy]), fabs(roi_projection.y + roi_projection.height - cp[cy]));
        //  double dm = sqrt(dmx*dmx + dmy*dmy);
        
        //double dm = (cp[w] * 0.5);
        double dm = std::max(dmx, dmy);
        double dperc = dp / dm;

        //yInfo() << dmx << 1-dperc << 1+dperc;
        
        M = cv::getRotationMatrix2D(cv::Point(cp[cx], cp[cy]), 0, 1 - dperc);
        cv::warpAffine(image_projection, warps_n[z], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M = cv::getRotationMatrix2D(cv::Point(cp[cx], cp[cy]), 0, 1 + dperc);
        cv::warpAffine(image_projection, warps_p[z], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        //how to update the state
        //du = -(u-cx)/d * dz -> dz = du * d / -(c-cx) -> dz = dpix * d / dm
        //dv = -(v-cy)/d * dz
        states_p[z][z] += dp * d / dm;
        states_n[z][z] -= dp * d / dm;

        scores_p[z] = similarity_score(obs, warps_p[z]);
        scores_n[z] = similarity_score(obs, warps_n[z]);
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
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            image_projection.copyTo(roi);
            score_overlay(score_projection, roi);
        }

        if(!warps_n[x].empty()) {
            col = 0; row = 1;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[x].copyTo(roi);
            score_overlay(scores_n[x], roi);
        }

        if(!warps_p[x].empty()) {
            col = 2; row = 1;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[x].copyTo(roi);
            score_overlay(scores_p[x], roi);
        }

        if(!warps_n[y].empty()) {
            col = 1; row = 0;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[y].copyTo(roi);
            score_overlay(scores_n[y], roi);
        }

        if(!warps_p[y].empty()) {
            col = 1; row = 2;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[y].copyTo(roi);
            score_overlay(scores_p[y], roi);
        }

        if(!warps_n[z].empty()) {
            col = 0; row = 0;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_n[z].copyTo(roi);
            score_overlay(scores_n[z], roi);
        }

        if(!warps_p[z].empty()) {
            col = 2; row = 2;
            cv::Mat roi = joined(cv::Rect(cp[w] * col, cp[h] * row, cp[w], cp[h]));
            warps_p[z].copyTo(roi);
            score_overlay(scores_p[z], roi);
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