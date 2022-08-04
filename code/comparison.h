#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat process_projected(cv::Mat projected)
{
    static cv::Mat canny_img, f, pos_hat, neg_hat;
    cv::Canny(projected, canny_img, 40, 40*3, 3);
    
    canny_img.convertTo(f, CV_32F);
    cv::GaussianBlur(f, pos_hat, cv::Size(21, 21), 0);
    cv::GaussianBlur(f, neg_hat, cv::Size(41, 41), 0);

    cv::Mat grey = cv::Mat::zeros(canny_img.size(), CV_32F);
    grey -= neg_hat;
    grey += pos_hat;

    double minval, maxval;
    cv::minMaxLoc(grey, &minval, &maxval);
    maxval = 2*std::max(fabs(minval), fabs(maxval));
    grey /= maxval;

    return grey;
}

cv::Mat process_eros(cv::Mat eros_img)
{
    static cv::Mat eros_blurred, eros_f;
    cv::GaussianBlur(eros_img, eros_blurred, cv::Size(7, 7), 0);
    eros_blurred.convertTo(eros_f, CV_32F);
    cv::normalize(eros_f, eros_f, 0.0, 1.0, cv::NORM_MINMAX);

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

void calculate_state_deltas()
{
    //from the interaction matrix and current depth calculate the desired state
    //change to move the image by approximately 1 pixel.

    //[du dv] =
    //[ fx/d, 0, -(u-cx)/d, -(u-cx)(v-cy)/fy, fx*fx+(u-cx)(u-cx)/fx, -(v-cy)fx/fy
    //  0, fy/d, -(v-cy)/d, -(fy*fy)-(v-cy)(v-cy)/fy, (u-cx)(v-cy)/fx, (u-cx)fy/fx] *

    // [dx, dy, dz, dalpha, dbeta, dgamma]

    // where d = sqrt(x^2 + y^2 + z^2)

    // MAX(u-cx) = mu
    // MAX(v-cy) = mv

    std::vector<double> dstate(6);

    //du = fx/d * dx
    //dx = du * d / fx;
    
    //dv = fy/d * dy
    //dy = dv * d / fy
    
    //du = -mu/d * dz
    //dv = -mv/d * dz
    //dz = MAX(-du*d/mu, -dv*d/mv) careful of negatives with max

    //



}

class predictions {

private:
    enum{x,y,z,a,b,c};
    enum{w,h,cx,cy,fx,fy};
    std::array<double, 6> cp;

    std::array<double, 7> state_current;
    double d;

    std::array<double, 7> state_projection;
    cv::Mat image_projection;
    double score_projection;

    std::array<cv::Mat, 6> warps_p;
    std::array<cv::Mat, 6> warps_n; 
    std::array<std::array<double, 7>, 6> states_p;
    std::array<std::array<double, 7>, 6> states_n;
    std::array<double, 6> scores_p;
    std::array<double, 6> scores_n;

public:
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
        d = sqrt(state[0] * state[0] + state[1] * state[1] + state[2] * state[2]);

        for(auto &s : states_p)
            s = state;

        for(auto &s : states_n)
            s = state;

    }

    void predict_x() 
    {
        // we want to move the image by 1 in the x axis
        static int du = 12;
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 1, 0, 1, 0);

        M.at<double>(0, 2) = du;
        cv::warpAffine(image_projection, warps_p[x], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M.at<double>(0, 2) = -du;
        cv::warpAffine(image_projection, warps_n[x], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        //yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[x][x] += (du * d /cp[fx]);
        states_n[x][x] -= (du * d /cp[fx]);

        //state[0] += 1 * d / cp[fx];
    }

    void predict_y() 
    {
        // we want to move the image by 1 in the x axis
        static cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 1);
        static int dv = 12;

        M.at<double>(1, 2) = dv;
        cv::warpAffine(image_projection, warps_p[y], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        M.at<double>(1, 2) = -dv;
        cv::warpAffine(image_projection, warps_n[y], M, image_projection.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // calculate the state change given interactive matrix
        // dx = du * d / fx
        // yInfo() << (8 * d /cp[fx]) *0.001;
        states_p[y][y] += (dv * d / cp[fy]);
        states_n[y][y] -= (dv * d / cp[fy]);

        // state[0] += 1 * d / cp[fx];
    }

    std::array<double, 7> score(const cv::Mat &obs)
    {

        score_projection = similarity_score(obs, image_projection) + 10;
        std::array<double, 7> best_state = state_projection;
        double best_score = score_projection;

        scores_p[x] = similarity_score(obs, warps_p[x]);
        if(scores_p[x] > best_score) {
            best_score = scores_p[x];
            best_state = (states_p[x]);
        }

        scores_n[x] = similarity_score(obs, warps_n[x]);
        if(scores_n[x] > best_score) {
            best_score = scores_n[x];
            best_state = (states_n[x]);
        }

        scores_p[y] = similarity_score(obs, warps_p[y]);
        if (scores_p[y] > best_score) {
            best_score = scores_p[y];
            best_state = (states_p[y]);
        }

        scores_n[y] = similarity_score(obs, warps_n[y]);
        if (scores_n[y] > best_score) {
            best_score = scores_n[y];
            best_state = (states_n[y]);
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

    cv::Mat create_visualisation() {

        score_overlay(score_projection, image_projection);
        score_overlay(scores_p[x], warps_p[x]);
        score_overlay(scores_n[x], warps_n[x]);

        cv::Mat joined(cp[h], cp[w]*3, CV_32F);
        warps_n[x].copyTo(joined(cv::Rect(0, 0, cp[w], cp[h])));
        image_projection.copyTo(joined(cv::Rect(cp[w], 0, cp[w], cp[h])));
        warps_p[x].copyTo(joined(cv::Rect(cp[w]*2, 0, cp[w], cp[h])));
        
        

        return joined;



    }
};



void predict(std::vector<double>& state, cv::Mat& projection, int axis, double delta)
{
    //get three random points in the image (given projection image size)

    //calculate the three warped positions given the interaction matrix

    //calculate the warped affine and apply the wap to the image

    //convert the state to eurler, add the delta, convert back to quaternion



}