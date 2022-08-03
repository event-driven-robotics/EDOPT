#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
using namespace Metavision;

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>

#include <yarp/os/all.h>
using namespace yarp::os;

#include <event-driven/core.h>
#include <event-driven/algs.h>
#include <event-driven/vis.h>

class EROSasynch
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

class tracker : public yarp::os::RFModule 
{
private:

    EROSasynch eros_handler;
    cv::Size res;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {

        double bias_sens = rf.check("s", Value(0.4)).asFloat64();
        double cam_filter = rf.check("f", Value(0.01)).asFloat64();

        if(!eros_handler.start(bias_sens, cam_filter)) 
        {
            return false;
        }


        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", eros_handler.res);

        return true;
    }

    double getPeriod() override
    {
        return 0.1;
    }
    bool updateModule() override
    {
        static cv::Mat mysurf;
        eros_handler.eros.getSurface().copyTo(mysurf);
        cv::imshow("EROS", mysurf);
        cv::waitKey(1);
        return true;
    }
    // bool interruptModule() override
    // {
    // }
    // bool close() override
    // {
    // }

};

int main(int argc, char* argv[])
{
    tracker my_tracker;
    ResourceFinder rf;
    rf.configure(argc, argv);
    
    return my_tracker.runModule(rf);
}