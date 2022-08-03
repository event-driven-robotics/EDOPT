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