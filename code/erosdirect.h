#pragma once

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
using namespace Metavision;
#include <yarp/os/all.h>
using namespace yarp::os;
#include <event-driven/algs.h>
#include <event-driven/vis.h>

#include "erosplus.h"

class EROSdirect
{
public:
    Metavision::Camera cam;
    ev::EROS eros;
    ev::vNoiseFilter filter;
    cv::Size res;

    bool start(double sensitivity, double filter_value)
    {
        if(sensitivity < 0.0 || sensitivity > 1.0)
        {
            yError() << "sensitivity 0 < s < 1";
            return false;
        }

        try {
            cam = Camera::from_first_available();
            I_LL_Biases* bias_control = cam.biases().get_facility();  
            int diff_on  = (66 - 350) * sensitivity + 650 - 66;
            int diff_off = (66 + 200) * sensitivity + 100 - 66;
            bias_control->set("bias_diff_on", diff_on);
            bias_control->set("bias_diff_off", diff_off);
        } catch(const std::exception &e) {
            yError() << "no camera :(";
            return false;
        }

        const Metavision::Geometry &geo = cam.geometry();
        yInfo() << "[" << geo.width() << "x" << geo.height() << "]";

        res =  cv::Size(geo.width(), geo.height());
        eros.init(res.width, res.height, 7, 0.3);
        filter.use_temporal_filter(filter_value);
        filter.initialise(res.width, res.height);

        cam.cd().add_callback([this](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            this->erosUpdate(ev_begin, ev_end);
        });

        if (!cam.start()) {
            yError() << "Could not start the camera";
            return false;
        }

        return true;
    }

    void stop()
    {
        cam.stop();
    }

    void erosUpdate(const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) 
    {
        double t = yarp::os::Time::now();
        for(auto &v = ev_begin; v != ev_end; ++v) 
        {
            if(filter.check(v->x, v->y, v->p, t))
                eros.update(v->x, v->y);
        }
    }

};

class EROSfromYARP
{
public:

    ev::window<ev::AE> input_port;
    ev::EROS eros;
    std::thread eros_worker;
    double tic{-1};
    cv::Mat event_image;

    void erosUpdate() 
    {
        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readAll(true);
            tic = my_info.timestamp;
            for(auto &v : input_port) {
                eros.update(v.x, v.y);
                if(v.p)
                    event_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(255, 255, 255);
                else 
                    event_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(255, 0, 0);
                
            }
        }
    }

public:
    bool start(cv::Size resolution, std::string sourcename, std::string portname, int k = 5, double d = 0.3)
    {
        eros.init(resolution.width, resolution.height, k, d);
        event_image = cv::Mat(resolution, CV_8UC3, cv::Vec3b(0, 0, 0));

        if (!input_port.open(portname))
            return false;
        yarp::os::Network::connect(sourcename, portname, "fast_tcp");

        eros_worker = std::thread([this]{erosUpdate();});
        return true;
    }

    void stop()
    {
        input_port.stop();
        eros_worker.join();
    }

};

class ARESfromYARP
{
public:

    ev::window<ev::AE> input_port;
    erosplus eros;
    std::thread eros_worker;
    double tic{-1};
    cv::Mat event_image;

    void erosUpdate() 
    {
        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readAll(true);
            tic = my_info.timestamp;
            for(auto &v : input_port)
                eros.update(v.x, v.y);
        }
    }

public:
    bool start(cv::Size resolution, std::string sourcename, std::string portname, int k = 5, double d = 0.3)
    {
        eros.init(resolution.width, resolution.height, 7, 0.05, 0.003);
        event_image = cv::Mat(resolution, CV_8UC3, cv::Vec3b(0, 0, 0));

        if (!input_port.open(portname))
            return false;
        yarp::os::Network::connect(sourcename, portname, "fast_tcp");

        eros_worker = std::thread([this]{erosUpdate();});
        return true;
    }

    void stop()
    {
        input_port.stop();
        eros_worker.join();
    }

};