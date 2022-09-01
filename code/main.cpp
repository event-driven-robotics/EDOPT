

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>

#include <yarp/os/all.h>
using namespace yarp::os;

#include "erosdirect.h"
#include "projection.h"
#include "comparison.h"
#include "image_processing.h"



class tracker : public yarp::os::RFModule 
{
private:

    std::thread worker;
    //EROSdirect eros_handler;
    EROSfromYARP eros_handler;

    imageProcessing img_handler;

    cv::Size img_size;

    predictions warp_handler;
    std::array<double, 6> intrinsics;

    //std::array<double, 7> default_state = {0, 0, 0.92, 0, 0, 0.7071068, 0.7071068};
    std::array<double, 7> initial_state, camera_pose, state;
    std::deque< std::array<double, 8> > data_to_save;

    SICAD* si_cad;

    cv::Mat proj_rgb, eros_u, proj_32f;
    double toc_eros{0}, toc_proj{0}, toc_projproc{0}, toc_warp{0};
    int toc_count{0};
    bool step{false};
    double period{0.1};

    std::ofstream fs;
    std::string file_name;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        setName(rf.check("name", Value("/ekom")).asString().c_str());
        double bias_sens = rf.check("s", Value(0.6)).asFloat64();
        double cam_filter = rf.check("f", Value(0.1)).asFloat64();

        yarp::os::Bottle& intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Wrong .ini file or [CAMERA_CALIBRATION] not present. Deal breaker.";
            return false;
        }

        intrinsics[0] = intrinsic_parameters.find("w").asInt32();
        intrinsics[1] = intrinsic_parameters.find("h").asInt32();
        intrinsics[2] = intrinsic_parameters.find("cx").asFloat32();
        intrinsics[3] = intrinsic_parameters.find("cy").asFloat32();
        intrinsics[4] = intrinsic_parameters.find("fx").asFloat32();
        intrinsics[5] = intrinsic_parameters.find("fy").asFloat32();

        if(!loadPose(rf, "object_pose", initial_state)) return false;
        if(!loadPose(rf, "camera_pose", camera_pose)) return false;
        state = initial_state;
        img_size = cv::Size(intrinsics[0], intrinsics[1]);

        if(!Network::checkNetwork(1.0)) {
            yError() << "could not connect to YARP";
            return false;
        }

        if (!eros_handler.start(img_size, "/atis3/AE:o", getName("/AE:fi"))) {
            yError() << "could not open the YARP eros handler";
            return false;
        }

        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        // if(!eros_handler.start(bias_sens, cam_filter)) 
        // {
        //     return false;
        // }

        // if(img_size.width != intrinsics[0] || img_size.height != intrinsics[1]) 
        // {
        //     yError() << "Provided camera parameters don't match data";
        //     return false;
        // }

        

        int rescale_size = 100;
        int blur = rescale_size / 20;
        double dp =  1;//+rescale_size / 100;
        warp_handler.initialise(intrinsics, cv::Size(rescale_size, rescale_size), blur);
        warp_handler.create_Ms(dp);

        img_handler.initialise(rescale_size, blur);

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", img_size);
        cv::moveWindow("EROS", 0, 0);

        eros_u = cv::Mat::zeros(img_size, CV_8U);
        proj_rgb = cv::Mat::zeros(img_size, CV_8UC3);
        proj_32f = cv::Mat::zeros(img_size, CV_32F);

        cv::namedWindow("Translations", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Translations", img_size);
        cv::moveWindow("Translations", 0, 540);

        cv::namedWindow("Rotations", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Rotations", img_size);
        cv::moveWindow("Rotations", 700, 540);

        //quaternion_test();
        //quaternion_test_camera();

        worker = std::thread([this]{sequential_loop();});

        if (rf.check("file")) {
            fs.open(rf.find("file").asString());
            if (!fs.is_open()) {
                yError() << "Could not open output file"
                         << rf.find("file").asString();
                return false;
            }
        }

        return true;
    }

    double getPeriod() override
    {
        return period;
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
            warp_handler.set_current(initial_state);
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

        if (toc_count) {
            yInfo() << (int)(toc_eros / toc_count) << "\t"
                    << (int)(toc_proj / toc_count) << "\t"
                    << (int)(toc_projproc / toc_count) << "\t"
                    << (int)(toc_warp / toc_count) << "\t"
                    << (int) toc_count / period << "Hz";
            toc_count = toc_warp = toc_projproc = toc_proj = toc_eros = 0;
        }
        return true;
    }

    void projection_loop()
    {
        //pause the warp_loop() 

        //set the current image
        warp_handler.projection.img_warp = img_handler.proc_proj;

        //warp the current projection based on the warp list
        // and clear the list
        warp_handler.warp_by_history(warp_handler.projection.img_warp);

        //copy over the region of interest
        img_handler.set_obs_rois_from_projected();
        warp_handler.scale = img_handler.scale;

        //set the current state
        state = warp_handler.state_current;

        //unpause the warp_loop()

        //project the current state
        complexProjection(si_cad, camera_pose, state, proj_rgb);

        //get the ROI of the current state
        img_handler.set_projection_rois(proj_rgb);
        //process the projection
        img_handler.setProcProj(proj_rgb);

        
    }

    void warp_loop()
    {
        //perform warps
        warp_handler.make_predictive_warps();

        //get the current EROS
        eros_handler.eros.getSurface().copyTo(eros_u);
        img_handler.setProcObs(eros_u);

        //perform the comparison
        warp_handler.score_predictive_warps();

        //update the state
        if (step) {
            // warp_handler.update_from_max();
            // warp_handler.update_all_possible();
            warp_handler.update_heuristically();
            //state = warp_handler.state_current;
            step = true;
        }

        //yield to projection loop if needed.

        


    }

    void sequential_loop()
    {
        double dataset_time = -1;
        warp_handler.set_current(state);

        while (!isStopping()) {

            double dtic = Time::now();

            if (!complexProjection(si_cad, camera_pose, state, proj_rgb)) {
                yError() << "Could not perform projection";
                return;
            }

            img_handler.set_projection_rois(proj_rgb);
            img_handler.set_obs_rois_from_projected();
            warp_handler.scale = img_handler.scale;
            //warp_handler.extract_rois(proj_rgb);
            double dtoc_proj = Time::now();
            
            img_handler.setProcProj(proj_rgb);
            //proj_32f = warp_handler.extract_projection(proj_rgb);
            //warp_handler.set_projection(proj_32f);
            warp_handler.projection.img_warp = img_handler.proc_proj;
            warp_handler.make_predictive_warps();
            double dtoc_projproc = Time::now();

            eros_handler.eros.getSurface().copyTo(eros_u);
            dataset_time = eros_handler.tic;
            img_handler.setProcObs(eros_u);
            warp_handler.proc_obs = img_handler.proc_obs;
            //warp_handler.set_observation(eros_u);
            double dtoc_eros = Time::now();

            warp_handler.score_predictive_warps();
            
            
            if(step) {
                //warp_handler.update_from_max();
                //warp_handler.update_all_possible();
                warp_handler.update_heuristically();
                state = warp_handler.state_current;
                step = true;
            }
            
            double dtoc_warp = Time::now();
            //yInfo() << "update:"<< dtoc_eros - dtoc_warp;

            this->toc_eros += ((dtoc_eros - dtoc_projproc) * 1e6);
            this->toc_proj += ((dtoc_proj - dtic) * 1e6);
            this->toc_projproc += ((dtoc_projproc - dtoc_proj) * 1e6);
            this->toc_warp += ((dtoc_warp - dtoc_eros) * 1e6);
            this->toc_count++;
            if (fs.is_open() && dataset_time > 0) {
                data_to_save.push_back({dataset_time, state[0], state[1], state[2], state[3], state[4], state[5], state[6]});
            }
        }
    }

    bool quaternion_test(bool return_value = true)
    {
        double delta = 0.01;
        auto temp_state = initial_state;

        for(double th = 0.0; th > -M_PI_2; th-=delta)
        {
            if (!simpleProjection(si_cad, temp_state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            cv::waitKey(1);
            perform_rotation(temp_state, 2, delta);
        }

        for(double th = 0.0; th > -M_PI_2; th-=delta)
        {
            if (!simpleProjection(si_cad, temp_state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            cv::waitKey(1);
            perform_rotation(temp_state, 1, delta);
        }

        for(double th = 0.0; th > -M_PI_2; th-=delta)
        {
            if (!simpleProjection(si_cad, temp_state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            cv::waitKey(1);
            perform_rotation(temp_state, 0, delta);
        }

        return return_value;
    }

    bool quaternion_test_camera(bool return_value = true)
    {

        auto camera = camera_pose;
        double delta = 0.01;

        //yInfo() << "First Rotation";
        for(double th = 0.0; th < 2*M_PI; th+=delta)
        {
            if (!complexProjection(si_cad, camera, state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            if(cv::waitKey(1) == '\e') return false;
            perform_rotation(camera, 2, delta);
            // yInfo() << camera[3] << camera[4] << camera[5] << camera[6];
            // yInfo() << 2 << th;
        }

        //yInfo() << "Second Rotation";
        for(double th = 0.0; th < 2*M_PI; th+=delta)
        {
            if (!complexProjection(si_cad, camera, state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            if(cv::waitKey(1) == '\e') return false;
            perform_rotation(camera, 1, delta);
            // yInfo() << camera[3] << camera[4] << camera[5] << camera[6];
            // yInfo() << 1 << th;
        }

        //yInfo() << "Third Rotation";
        for(double th = 0.0; th < 2*M_PI; th+=delta)
        {
            if (!complexProjection(si_cad, camera, state, proj_rgb)) {
                yError() << "Could not perform projection";
                return false;
            }
            cv::imshow("qtest", proj_rgb);
            if(cv::waitKey(1) == '\e') return false;
            perform_rotation(camera, 0, delta);
            // yInfo() << camera[3] << camera[4] << camera[5] << camera[6];
            // yInfo() << 0 << th;
        }
        //yInfo()<< "Done";

        return return_value;
    }

    bool interruptModule() override
    {
        yInfo() << "interrupt caught";
        return true;
    }

    bool close() override
    {
        yInfo() << "waiting for eros handler ... ";
        eros_handler.stop();
        yInfo() << "waiting for workther thread ... ";
        worker.join();
        if(fs.is_open())
        {
            yInfo() << "Writing data ...";
            for(auto i : data_to_save)
                fs << i[0] << ", " << i[1] << ", " << i[2] << ", " << i[3] << ", " << i[4] << ", " << i[5] << ", " << i[6] << ", " << i[7] << std::endl;
            fs.close();
            yInfo() << "Finished Writing data ...";
        }
        yInfo() << "close function finished";
        return true;
    }

};

int main(int argc, char* argv[])
{
    tracker my_tracker;
    ResourceFinder rf;
    rf.setDefaultConfigFile("/usr/local/src/object-track-6dof/configCAR.ini");
    rf.configure(argc, argv);
    
    return my_tracker.runModule(rf);
}
