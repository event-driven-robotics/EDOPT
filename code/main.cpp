

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>

#include <yarp/os/all.h>
using namespace yarp::os;

#include "erosdirect.h"
#include "projection.h"
#include "comparison.h"



class tracker : public yarp::os::RFModule 
{
private:

    EROSdirect eros_handler;

    predictions warp_handler;
    std::array<double, 6> intrinsics;

    std::array<double, 7> state = {0, 0, -100, 0.0, -1.0, 0.0, 0};
    SICAD* si_cad;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {

        double bias_sens = rf.check("s", Value(0.4)).asFloat64();
        double cam_filter = rf.check("f", Value(0.01)).asFloat64();

        yarp::os::Bottle& intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Could not load camera parameters";
            return false;
        }
        intrinsics[0] = intrinsic_parameters.find("w").asInt();
        intrinsics[1] = intrinsic_parameters.find("h").asInt();
        intrinsics[2] = intrinsic_parameters.find("cx").asDouble();
        intrinsics[3] = intrinsic_parameters.find("cy").asDouble();
        intrinsics[4] = intrinsic_parameters.find("fx").asDouble();
        intrinsics[5] = intrinsic_parameters.find("fy").asDouble();

        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        
        if(!eros_handler.start(bias_sens, cam_filter)) 
        {
            return false;
        }

        if(eros_handler.res.width != intrinsics[0] || eros_handler.res.height != intrinsics[1]) 
        {
            yError() << "Provided camera parameters don't match data";
            return false;
        }

        warp_handler.set_intrinsics(intrinsics);

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", eros_handler.res);

        cv::namedWindow("Projection", cv::WINDOW_AUTOSIZE);
        //cv::resizeWindow("Projection", eros_handler.res);

        return true;
    }

    double getPeriod() override
    {
        return 0;
    }
    bool updateModule() override
    {
        cv::Mat eros_f = process_eros(eros_handler.eros.getSurface()); 
        //cv::Mat eros_f = cv::Mat::zeros(intrinsics[1], intrinsics[0], CV_32F);

        cv::Mat projected_image;
        //yInfo() << state[0] << state[1] << state[2] << state[3] << state[4] << state[5] << state[6];
        Superimpose::ModelPose pose = quaternion_to_axisangle(state);
        if (!simpleProjection(si_cad, pose, projected_image)) {
            yError() << "Could not perform projection";
            return false;
        }
        cv::Mat proj_f = process_projected(projected_image);

        warp_handler.set_projection(state, proj_f);
        warp_handler.predict_x();
        warp_handler.predict_y();
        state = warp_handler.score(eros_f);
        cv::Mat vis = warp_handler.create_visualisation();

        //yInfo() << similarity_score(eros_f, proj_f);
        cv::imshow("EROS", make_visualisation(eros_f, proj_f));
        cv::imshow("Projection", vis+0.5);
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