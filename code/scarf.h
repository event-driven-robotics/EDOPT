#pragma once

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
using namespace Metavision;
#include <yarp/os/all.h>
using namespace yarp::os;
#include <event-driven/algs.h>
#include <event-driven/vis.h>

class SCARFdirect
{
public:
    Metavision::Camera cam;
    ev::SCARF scarf;
    ev::vNoiseFilter filter;
    cv::Size res;
    int block_size{14};
    double alpha{1.0};
    double c_factor{0.8};

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
        scarf.initialise(res, block_size, alpha, c_factor);
        filter.use_temporal_filter(filter_value);
        filter.initialise(res.width, res.height);

        cam.cd().add_callback([this](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            this->scarfUpdate(ev_begin, ev_end);
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

    void scarfUpdate(const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) 
    {
        double t = yarp::os::Time::now();
        for(auto &v = ev_begin; v != ev_end; ++v) 
        {
            if(filter.check(v->x, v->y, v->p, t))
                scarf.update(v->x, v->y, v->p);
        }
    }

};

class SCARFfromYARP
{
public:

    ev::window<ev::AE> input_port;
    ev::SCARF scarf;
    std::thread scarf_worker;
    double tic{-1};
    cv::Mat event_image;
    int block_size;
    double alpha;
    double c_factor;


    void scarfUpdate() 
    {
        while (!input_port.isStopping()) {
            ev::info my_info = input_port.readAll(true);
            tic = my_info.timestamp;
            for(auto &v : input_port) {
                scarf.update(v.x, v.y, v.p);
                if(v.p)
                    event_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(255, 255, 255);
                else 
                    event_image.at<cv::Vec3b>(v.y, v.x) = cv::Vec3b(255, 0, 0);
                
            }
        }
    }

public:
    bool start(cv::Size resolution, std::string sourcename, std::string portname, int block_size = 14, double alpha = 1.0, double c_factor=0.8)
    {
        scarf.initialise(resolution, block_size, alpha, c_factor);
        event_image = cv::Mat(resolution, CV_8UC3, cv::Vec3b(0, 0, 0));

        if (!input_port.open(portname))
            return false;
        yarp::os::Network::connect(sourcename, portname, "fast_tcp");

        scarf_worker = std::thread([this]{scarfUpdate();});
        return true;
    }

    void stop()
    {
        input_port.stop();
        scarf_worker.join();
    }

};