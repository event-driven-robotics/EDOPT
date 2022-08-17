

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

    std::thread worker;
    EROSdirect eros_handler;

    predictions warp_handler;
    std::array<double, 6> intrinsics;

    std::array<double, 7> default_state = {0, 0, -300, 0.0, -1.0, 0.0, 0};
    std::array<double, 7> state;
    SICAD* si_cad;

    cv::Mat eros_f, proj_f;
    double tic, toc_eros, toc_proj, toc_projproc, toc_warp;
    bool step{false};

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {

        double bias_sens = rf.check("s", Value(0.4)).asFloat64();
        double cam_filter = rf.check("f", Value(0.01)).asFloat64();

        yarp::os::Bottle& intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Wrong .ini file or [CAMERA_CALIBRATION] not present. Deal breaker.";
            return false;
        }

        intrinsics[0] = intrinsic_parameters.find("w").asInt();
        intrinsics[1] = intrinsic_parameters.find("h").asInt();
        intrinsics[2] = intrinsic_parameters.find("cx").asDouble();
        intrinsics[3] = intrinsic_parameters.find("cy").asDouble();
        intrinsics[4] = intrinsic_parameters.find("fx").asDouble();
        intrinsics[5] = intrinsic_parameters.find("fy").asDouble();

        state = default_state;

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
        cv::moveWindow("EROS", 1920, 0);
        eros_f = cv::Mat::zeros(eros_handler.res, CV_32F);
        proj_f = cv::Mat::zeros(eros_handler.res, CV_32F);

        worker = std::thread([this]{main_loop();});

        cv::namedWindow("Projection", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Projection", eros_handler.res);
        cv::moveWindow("Projection", 1920, 540);

        return true;
    }

    double getPeriod() override
    {
        return 0.1;
    }

    bool updateModule() override
    {
        static cv::Mat vis;
        vis = make_visualisation(eros_f, proj_f);
        static cv::Mat warps = cv::Mat::zeros(100, 100, CV_8U);
        warps = warp_handler.create_translation_visualisation();
        cv::imshow("EROS", vis);
        cv::imshow("Projection", warps+0.5);
        int c = cv::waitKey(1);
        if (c == 32)
            state = default_state;
        if (c == 'g')
            step = true;
        if(c == 27)
            return false;
        yInfo() << (int)toc_eros << "\t" 
                << (int)toc_proj << "\t"
                << (int)toc_projproc << "\t"
                << (int)toc_warp;
        return true;
    }



    void main_loop()
    {
        int dp = 4;
        int blur = 10;
        while (!isStopping()) {
            // double tic = Time::now();
            double tic = Time::now();
            eros_f = process_eros(eros_handler.eros.getSurface());
            double toc_eros = Time::now();

        
            static cv::Mat projected_image = cv::Mat::zeros(eros_f.size(), CV_8UC3);
            Superimpose::ModelPose pose = quaternion_to_axisangle(state);
            if (!simpleProjection(si_cad, pose, projected_image)) {
                yError() << "Could not perform projection";
                return;

            }
            double toc_proj = Time::now();
            
            proj_f = process_projected(projected_image, blur);

            double toc_projproc = Time::now();

            warp_handler.extract_roi(projected_image);
            warp_handler.set_current(state);
            warp_handler.set_projection(state, proj_f);
            warp_handler.reset_comparison(eros_f);
            warp_handler.compare_to_warp_x(eros_f, dp);
            warp_handler.compare_to_warp_y(eros_f, dp);
            warp_handler.compare_to_warp_z(eros_f, dp);
            if(step) {
                state = warp_handler.next_best();
                step = true;
            }
            double toc_warp = Time::now();
            //rate = 1.0 / (Time::now() - tic);

            this->toc_eros= ((toc_eros - tic) * 10e3);
            this->toc_proj= ((toc_proj - toc_eros) * 10e3);
            this->toc_projproc = ((toc_projproc - toc_proj) * 10e3);
            this->toc_warp= ((toc_warp - toc_projproc) * 10e3);
        }
    }

    // bool interruptModule() override
    // {
    // }
    bool close() override
    {
        worker.join();
        return true;
    }

};

int main(int argc, char* argv[])
{
    tracker my_tracker;
    ResourceFinder rf;
    rf.setDefaultConfigFile("/usr/local/src/object-track-6dof/");
    rf.configure(argc, argv);
    
    return my_tracker.runModule(rf);
}