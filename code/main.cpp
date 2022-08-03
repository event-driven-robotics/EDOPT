

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>

#include <yarp/os/all.h>
using namespace yarp::os;

#include "erosdirect.h"
#include "projection.h"

class tracker : public yarp::os::RFModule 
{
private:

    EROSdirect eros_handler;

    std::vector<double> state = {0, 0, -200, 0.0, -1.0, 0.0, 0};
    SICAD* si_cad;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {

        double bias_sens = rf.check("s", Value(0.4)).asFloat64();
        double cam_filter = rf.check("f", Value(0.01)).asFloat64();


        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        
        if(!eros_handler.start(bias_sens, cam_filter)) 
        {
            return false;
        }

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", eros_handler.res);

        cv::namedWindow("Projection", cv::WINDOW_NORMAL);
        cv::resizeWindow("Projection", eros_handler.res);

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
        

        cv::Mat projected_image;
        Superimpose::ModelPose pose = quaternion_to_axisangle(state);
        if (!simpleProjection(si_cad, pose, projected_image)) {
            yError() << "Could not perform projection";
            return false;
        }
        cv::imshow("Projection", projected_image);
        state[6] += 0.1;
        normalise_quaternion(state);


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