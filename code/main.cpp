

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

    cv::Size img_size;

    predictions warp_handler;
    std::array<double, 6> intrinsics;

    std::array<double, 7> default_state = {0, 0, -300, 0.0, -1.0, 0.0, 0};
    std::array<double, 7> state;
    SICAD* si_cad;

    cv::Mat proj_rgb, eros_u;
    double tic, toc_eros, toc_proj, toc_projproc, toc_warp;
    bool step{false};

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {

        double bias_sens = rf.check("s", Value(0.5)).asFloat64();
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

        img_size = eros_handler.res;

        if(img_size.width != intrinsics[0] || img_size.height != intrinsics[1]) 
        {
            yError() << "Provided camera parameters don't match data";
            return false;
        }

        int rescale_size = 120;
        int blur = rescale_size / 10;
        double dp =  1;//+rescale_size / 100;
        warp_handler.initialise(intrinsics, cv::Size(rescale_size, rescale_size), blur);
        warp_handler.create_Ms(dp);

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", img_size);
        cv::moveWindow("EROS", 0, 0);

        eros_u = cv::Mat::zeros(img_size, CV_8U);
        proj_rgb = cv::Mat::zeros(img_size, CV_8UC3);

        
        worker = std::thread([this]{main_loop();});

        cv::namedWindow("Translations", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Translations", img_size);
        cv::moveWindow("Translations", 0, 540);

        cv::namedWindow("Rotations", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Rotations", img_size);
        cv::moveWindow("Rotations", 700, 540);

        return true;
    }

    double getPeriod() override
    {
        return 0.1;
    }

    bool updateModule() override
    {
        static cv::Mat vis, proj_vis, eros_vis;
        //vis = warp_handler.make_visualisation(eros_u);
        proj_rgb.copyTo(proj_vis);
        cv::cvtColor(eros_u, eros_vis, cv::COLOR_GRAY2BGR);
        vis = proj_rgb*0.5 + eros_vis*0.5;
        static cv::Mat warps_t = cv::Mat::zeros(100, 100, CV_8U);
        warps_t = warp_handler.create_translation_visualisation();
        static cv::Mat warps_r = cv::Mat::zeros(100, 100, CV_8U);
        warps_r = warp_handler.create_rotation_visualisation();
        cv::imshow("EROS", vis);
        cv::imshow("Translations", warps_t+0.5);
        cv::imshow("Rotations", warps_r+0.5);
        int c = cv::waitKey(1);
        if (c == 32)
            warp_handler.set_current(default_state);
        if (c == 'g')
            step = true;
        if(c == 27) {
            stopModule();
            return false;
        }

        // yInfo() << cv::sum(cv::sum(warp_handler.warps[predictions::z].img_warp))[0]
        //         << cv::sum(cv::sum(warp_handler.projection.img_warp))[0];

        // std::array<double, 6> s = warp_handler.scores_p;
        // yInfo() << warp_handler.score_projection << s[0] << s[1] << s[2] << s[3] << s[4] << s[5];
        // s = warp_handler.scores_n;
        // yInfo() << warp_handler.score_projection << s[0] << s[1] << s[2] << s[3] << s[4] << s[5];
        // yInfo();

        // yInfo() << state[0] << state[1] << state[2] << state[3] << state[4]
        //         << state[5] << state[6];

        yInfo() << (int)toc_eros << "\t" 
                << (int)toc_proj << "\t"
                << (int)toc_projproc << "\t"
                << (int)toc_warp;
        return true;
    }

    void main_loop()
    {
        warp_handler.set_current(state);
        while (!isStopping()) {

            double tic = Time::now();

            Superimpose::ModelPose pose = quaternion_to_axisangle(state);
            if (!simpleProjection(si_cad, pose, proj_rgb)) {
                yError() << "Could not perform projection";
                return;
            }

            warp_handler.extract_rois(proj_rgb);
            double toc_proj = Time::now();
                        
            warp_handler.set_projection(state, proj_rgb);
            double toc_projproc = Time::now();

            eros_handler.eros.getSurface().copyTo(eros_u);
            
            warp_handler.set_observation(eros_u);
            double toc_eros = Time::now();
            
            warp_handler.compare_to_warp_x();
            warp_handler.compare_to_warp_y();
            warp_handler.compare_to_warp_z();
            warp_handler.compare_to_warp_a();
            warp_handler.compare_to_warp_b();
            warp_handler.compare_to_warp_c();
            
            if(step) {
                warp_handler.update_from_max();
                state = warp_handler.state_current;
                step = true;
            }
            double toc_warp = Time::now();

            this->toc_eros= ((toc_eros - toc_projproc) * 10e3);
            this->toc_proj= ((toc_proj - tic) * 10e3);
            this->toc_projproc = ((toc_projproc - toc_proj) * 10e3);
            this->toc_warp= ((toc_warp - toc_eros) * 10e3);
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