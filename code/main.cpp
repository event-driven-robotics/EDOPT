#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_cd.h>
using namespace Metavision;

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>

#include <yarp/os/all.h>
using namespace yarp::os;

#include <event-driven/core.h>
#include <event-driven/algs.h>

class tracker : public yarp::os::RFModule 
{
private:

    Metavision::Camera cam;
    ev::EROS eros;
    cv::Size res;


    void erosUpdate(const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end)
    {
        for(auto &v = ev_begin; v != ev_end; ++v) 
        {
            eros.update(v->x, v->y);
        }
    }

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        (void)rf;

        try {
            cam = Camera::from_first_available();
        } catch(const std::exception &e) {
            yError() << "no camera :(";
            return false;
        }

        const Metavision::Geometry &geo = cam.geometry();
        yInfo() << "[" << geo.width() << "x" << geo.height() << "]";

        res =  cv::Size(geo.width(), geo.height());
        eros.init(res.width, res.height, 7, 0.3);

        cam.cd().add_callback([this](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            this->erosUpdate(ev_begin, ev_end);
        });

        if(!cam.start()) {
            yError() << "Could not start the camera";
            return false;
        }

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", res);

        return true;
    }

    double getPeriod() override
    {
        return 0.1;
    }
    bool updateModule() override
    {
        static cv::Mat mysurf; 
        eros.getSurface().copyTo(mysurf);
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