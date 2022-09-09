#pragma once

#include <opencv2/opencv.hpp>

class imageProcessing
{
public:
    //given parameters
    int blur{10};
    cv::Size proc_size{cv::Size(100, 100)};
    int canny_thresh{40};
    double canny_scale{3};

    //internal variables
    //for the projection
    cv::Rect img_roi;  //roi on the image (square)
    cv::Rect proc_roi; //roi within the proc_proj
    
    //for the observation (copies for threading purposes)
    cv::Rect o_img_roi;  //roi within the image 
    cv::Rect o_proc_roi; //roi within the proc_obs

    //output
    double scale{1.0}; //final scale between process and image roi
    cv::Mat proc_proj; //final projection used to compare to
    cv::Mat proc_obs;  //final eros to compare to 

    void make_template(const cv::Mat &input, cv::Mat &output) {
        static cv::Mat canny_img, f, pos_hat, neg_hat;
        static cv::Size pblur(blur, blur);
        static cv::Size nblur(2*blur-1, 2*blur-1);
        static double minval, maxval;
        
        cv::Canny(input, canny_img, canny_thresh, canny_thresh*canny_scale, 3);
        canny_img.convertTo(f, CV_32F);

        cv::GaussianBlur(f, pos_hat, pblur, 0);
        cv::GaussianBlur(f, neg_hat, nblur, 0);
        output = pos_hat - neg_hat;
        cv::minMaxLoc(output, &minval, &maxval);
        double scale_factor = 1.0 / (2 * std::max(fabs(minval), fabs(maxval)));
        output *= scale_factor;
    }

    void process_eros(const cv::Mat &input, cv::Mat &output) {
        static cv::Mat eros_blurred, eros_f;
        cv::GaussianBlur(input, eros_blurred, cv::Size(7, 7), 0);
        eros_blurred.convertTo(output, CV_32F, 0.003921569);
        //cv::normalize(eros_f, eros_fn, 0.0, 1.0, cv::NORM_MINMAX);
    }

public:

    void initialise(int process_size, int blur_size, int canny_thresh, double canny_scale)
    {
        this->canny_thresh = canny_thresh;
        this->canny_scale = canny_scale;
        blur = blur_size % 2 ? blur_size : blur_size + 1;
        proc_size = cv::Size(process_size, process_size);
        proc_proj = cv::Mat(proc_size, CV_32F);
        proc_obs  = cv::Mat(proc_size, CV_32F);
        o_img_roi = o_proc_roi = proc_roi = img_roi = cv::Rect(cv::Point(0, 0), proc_size);
    }

    void set_projection_rois(const cv::Mat &projected, int buffer = 20) {

        static cv::Rect full_roi = cv::Rect(cv::Point(0, 0), projected.size());

        // convert to grey
        static cv::Mat grey;
        cv::cvtColor(projected, grey, cv::COLOR_BGR2GRAY);

        // find the bounding rectangle and add some buffer
        img_roi = cv::boundingRect(grey);
        img_roi.x -= buffer;
        img_roi.y -= buffer;
        img_roi.width += buffer * 2;
        img_roi.height += buffer * 2;

        // limit the roi to the image space.
        img_roi = img_roi & full_roi;

        // find the process rois and the scale factor
        if (img_roi.width >= img_roi.height) {
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

    void set_obs_rois_from_projected()
    {
        //projection  roi's are separate from eros roi's so that projection
        //rois can be updated in parallel, and copied over when thread safe
        o_img_roi = img_roi;
        o_proc_roi = proc_roi;
    }


    void setProcProj(const cv::Mat &image)
    {
        //the projection(roi) is resized to the process size and then processed
        static cv::Mat roi_rgb = cv::Mat::zeros(proc_size, CV_8UC3);
        roi_rgb = 0;
        cv::resize(image(img_roi), roi_rgb(proc_roi), proc_roi.size(), 0, 0, cv::INTER_CUBIC);
        make_template(roi_rgb, proc_proj);
    }

    void setProcObs(const cv::Mat &image)
    {
        //eros(roi) is processed as full image size and then resized
        //otherwise it has too many artefacts
        static cv::Mat roi_32f;// = cv::Mat::zeros(proc_size, CV_32F);
        proc_obs = 0;
        process_eros(image(o_img_roi), roi_32f);
        cv::resize(roi_32f, proc_obs(o_proc_roi), o_proc_roi.size(), 0, 0, cv::INTER_CUBIC);
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