

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <mutex>
#include <sstream>

#include <yarp/os/all.h>
using namespace yarp::os;

#include "erosdirect.h"
#include "projection.h"
#include "comparison.h"
#include "image_processing.h"

//__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia 


class tracker : public yarp::os::RFModule 
{
private:

    //parameters
    int proc_size{100};
    int canny_thresh{40}; double canny_scale{3.0};
    int eros_k{7}; double eros_d{0.7};
    bool dp2{false};
    bool parallel_method;
    bool run{true};
    double period{0.1};
    double render_scaler{1.0};
    int block_size; 
    double alpha, c_factor;

    //threads and mutex, handlers
    std::thread worker;
    std::thread proj_worker;
    std::thread warp_worker;
    std::mutex m;
    std::condition_variable signal;

    //handlers
    EROSfromYARP eros_handler;
    // SCARFfromYARP scarf_handler;
    //ARESfromYARP eros_handler;
    imageProcessing img_handler;
    warpManager warp_handler;

    // ev::SCARF scarf;
    ev::EROS eros;

    //internal variables
    SICAD* si_cad;
    cv::Size img_size;
    std::array<double, 6> intrinsics;
    std::array<double, 7> initial_state, camera_pose, state;
    cv::Mat proj_rgb, proj_32f;
    bool projection_available{false};

    //stats
    double toc_eros{0}, toc_proj{0}, toc_projproc{0}, toc_warp{0};
    int toc_count{0};
    int proj_count{0}, warp_count{0};
    int vis_type{0};

    //output
    std::ofstream fs;
    std::string file_name;
    std::deque< std::array<double, 8> > data_to_save;
    cv::VideoWriter vid_writer;

public:

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        setName(rf.check("name", Value("/ekom")).asString().c_str());
        double bias_sens = rf.check("s", Value(0.6)).asFloat64();
        double cam_filter = rf.check("f", Value(0.1)).asFloat64();

        proc_size = rf.check("proc_size", Value(100)).asInt32();
        canny_thresh = rf.check("canny_thresh", Value(40)).asInt32();
        canny_scale = rf.check("canny_scale", Value(3)).asFloat64();
        eros_k = rf.check("eros_k", Value(7)).asInt32();
        eros_d = rf.check("eros_d", Value(0.75)).asFloat64();
        block_size = rf.check("block_size", Value(11)).asInt32();//14
        alpha = rf.check("alpha", Value(2.0)).asFloat64(); //1.0
        c_factor = rf.check("c", Value(0.8)).asFloat64(); //0.8
        period = rf.check("period", Value(0.1)).asFloat64();
        dp2 = rf.check("dp2") && rf.check("dp2", Value(true)).asBool(); //default false
        // dp2 =true; 
        //run = rf.check("run") && !rf.find("run").asBool() ? false : true; //default true
        run = rf.check("run", Value(false)).asBool();
        parallel_method = rf.check("parallel") && rf.check("parallel", Value(true)).asBool(); // defaulat false
        render_scaler = rf.check("render_scaler", Value(1.0)).asFloat64();

        yarp::os::Bottle& intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Wrong .ini file or [CAMERA_CALIBRATION] not present. Deal breaker.";
            return false;
        }
        warp_handler.cam[warpManager::x] = intrinsic_parameters.find("w").asInt32()*render_scaler;
        warp_handler.cam[warpManager::y] = intrinsic_parameters.find("h").asInt32()*render_scaler;
        warp_handler.cam[warpManager::cx] = intrinsic_parameters.find("cx").asFloat32()*render_scaler;
        warp_handler.cam[warpManager::cy] = intrinsic_parameters.find("cy").asFloat32()*render_scaler;
        warp_handler.cam[warpManager::fx] = intrinsic_parameters.find("fx").asFloat32()*render_scaler;
        warp_handler.cam[warpManager::fy] = intrinsic_parameters.find("fy").asFloat32()*render_scaler;
        img_size = cv::Size(warp_handler.cam[warpManager::x], warp_handler.cam[warpManager::y]);

        if(!loadPose(rf, "object_pose", initial_state)) return false;
        if(!loadPose(rf, "camera_pose", camera_pose)) return false;
        state = initial_state;
        
        if(!Network::checkNetwork(1.0)) {
            yError() << "could not connect to YARP";
            return false;
        }

        // if (!scarf_handler.start(cv::Size(intrinsic_parameters.find("w").asInt32(), intrinsic_parameters.find("h").asInt32()), "/file/leftdvs:o", getName("/AE:i"), block_size, alpha, c_factor)) {
        //     yError() << "could not open the YARP eros handler";
        //     return false;
        // }

        if (!eros_handler.start(cv::Size(intrinsic_parameters.find("w").asInt32(), intrinsic_parameters.find("h").asInt32()), "/file/leftdvs:o", getName("/AE:i"), eros_k, eros_d)) {
            yError() << "could not open the YARP eros handler";
            return false;
        }

        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        int blur = proc_size / 20;
        double dp = 2;//+rescale_size / 100;
        warp_handler.initialise(proc_size, dp2);
        warp_handler.create_Ms(dp);
        warp_handler.set_current(state);
        img_handler.initialise(proc_size, blur, canny_thresh, canny_scale);

        proj_rgb = cv::Mat::zeros(img_size, CV_8UC1);
        proj_32f = cv::Mat::zeros(img_size, CV_32F);

        cv::namedWindow("EROS", cv::WINDOW_NORMAL);
        cv::resizeWindow("EROS", img_size);
        cv::moveWindow("EROS", 1920, 100);

        // cv::namedWindow("Translations", cv::WINDOW_AUTOSIZE);
        // cv::resizeWindow("Translations", img_size);
        // cv::moveWindow("Translations", 0, 540);

        // cv::namedWindow("Rotations", cv::WINDOW_AUTOSIZE);
        // cv::resizeWindow("Rotations", img_size);
        // cv::moveWindow("Rotations", 700, 540);

        //quaternion_test();
        //quaternion_test_camera();

        if(parallel_method) {
            proj_worker = std::thread([this]{projection_loop();});
            warp_worker = std::thread([this]{warp_loop();});
        } else {
            worker = std::thread([this]{sequential_loop();});
        }
        
        if (rf.check("file")) {
            std::string filename = rf.find("file").asString();
            fs.open(filename);
            if (!fs.is_open()) {
                yError() << "Could not open output file" << filename;
                return false;
            }
            vid_writer.open(filename + ".mp4", cv::VideoWriter::fourcc('H','2','6','4'), (int)(1.0/period), img_size, true);
            if (!vid_writer.isOpened()) {
                yError() << "Could not open output file" << filename << ".mp4";
                return false;
            }
        }


        yInfo() << "====== Configuration ======";
        yInfo() << "Camera Size:" << img_size.width << "x" << img_size.height;
        yInfo() << "Process re-sized:" << proc_size << "x" << proc_size << "(--proc_size <int>)";
        yInfo() << "EROS:" << eros_k << "," << eros_d << "(--eros_k <int> --eros_d <float>)";
        yInfo() << "Canny:" << canny_thresh << "," << canny_thresh<<"x"<<canny_scale << "(--canny_thresh <int> --canny_scale <float>)";
        if(dp2){yInfo()<<"ON: multi-size dp warps (--dp2)";}else{yInfo()<<"OFF: multi-size dp warps (--dp2)";}
        if(parallel_method){yInfo()<<"ON: threaded projections (--parallel)";}else{yInfo()<<"OFF: threaded projections (--parallel)";}
        if(!run) yInfo() << "WARNING: press G to start tracking (--run)";
        yInfo() << "===========================";

        return true;
    }



    double getPeriod() override
    {
        return period;
    }

    bool updateModule() override
    {
        static cv::Mat vis, proj_vis, scarf_vis, eros_vis;
        //vis = warp_handler.make_visualisation(eros_u);
        //proj_rgb.copyTo(proj_vis); 
        //img_handler.process_eros(eros_handler.eros.getSurface(), eros_vis);

        // cv::Mat img32 = scarf_handler.scarf.getSurface();
        // cv::Mat img8U;
        // img32.convertTo(img8U, CV_8U, 255);
        // cv::cvtColor(img8U, scarf_vis, cv::COLOR_GRAY2BGR);
        // cv::resize(scarf_vis, scarf_vis, img_size);
        // cv::cvtColor(proj_rgb, proj_vis, cv::COLOR_GRAY2BGR);
        // vis = proj_vis + scarf_vis;

        static cv::Mat eros_blurred1, eros_blurred2, eros_f, img32;
        cv::Mat eros8u = eros_handler.eros.getSurface();
        cv::medianBlur(eros8u, eros_blurred1, 3);
        cv::GaussianBlur(eros_blurred1, eros_blurred2, cv::Size(3, 3), 0);
        eros_blurred1.convertTo(img32, CV_32F, 0.003921569);

        cv::Mat img8U;
        img32.convertTo(img8U, CV_8U, 255);
        cv::cvtColor(img8U, eros_vis, cv::COLOR_GRAY2BGR);
        cv::resize(eros_vis, eros_vis, img_size);
        cv::cvtColor(proj_rgb, proj_vis, cv::COLOR_GRAY2BGR);
        vis = proj_vis + eros_vis;

        //cv::rectangle(vis, img_handler.img_roi, cv::Scalar(255, 255, 255));
        // static cv::Mat warps_t = cv::Mat::zeros(100, 100, CV_8U);
        // warps_t = warp_handler.create_translation_visualisation();
        // static cv::Mat warps_r = cv::Mat::zeros(100, 100, CV_8U);
        // warps_r = warp_handler.create_rotation_visualisation();
        //cv::flip(vis, vis, 1);
        
        std::string rate_string = "- Hz";
        std::stringstream ss; 
        ss.str();
        if(toc_count)
            ss << (int)toc_count / (period*10) << "Hz";
        
        cv::putText(vis, ss.str(), cv::Point(640-160, 480-10), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(200, 200, 200));

        cv::imshow("EROS", vis);
        //cv::imshow("Translations", warps_t+0.5);
        //cv::imshow("Rotations", warps_r+0.5);
        static bool stop_running = false;
        if(stop_running) {
            run = stop_running = false;
        }
        int c = cv::waitKey(1);
        if (c == 32) {
            state = initial_state;
            warp_handler.set_current(initial_state);
            stop_running = true;
        }
        // if (c == 'g')
        if (eros_handler.dt>0)
            run = true;
        if(c == 27) {
            stopModule();
            return false;
        }

        if(c == 'v')
            vis_type = (++vis_type % 2);

        // yInfo() << cv::sum(cv::sum(warp_handler.warps[predictions::z].img_warp))[0]
        //         << cv::sum(cv::sum(warp_handler.projection.img_warp))[0];

        // std::array<double, 6> s = warp_handler.scores_p;
        // yInfo() << warp_handler.score_projection << s[0] << s[1] << s[2] << s[3] << s[4] << s[5];
        // s = warp_handler.scores_n;
        // yInfo() << warp_handler.score_projection << s[0] << s[1] << s[2] << s[3] << s[4] << s[5];
         //yInfo();

        // yInfo() << state[0] << state[1] << state[2] << state[3] << state[4]
        //         << state[5] << state[6];
        if(vid_writer.isOpened() && eros_handler.tic > 0)
            vid_writer << vis;
        static int updated_divisor=0;
        if (updated_divisor++ % 10 == 0) {
            if (toc_count) {
                // yInfo() << (int)(toc_eros / toc_count) << "\t"
                //         << (int)(toc_proj / toc_count) << "\t"
                //         << (int)(toc_projproc / toc_count) << "\t"
                //         << (int)(toc_warp / toc_count) << "\t"
                //         << (int)toc_count / (period*10) << "Hz";
                toc_count = toc_warp = toc_projproc = toc_proj = toc_eros = 0;
            }
            if (warp_count || proj_count) {
                // yInfo() << (int)(warp_count / (period*10)) << "Hz" << (int)(proj_count / (10*period)) << "Hz";
                warp_count = proj_count = 0;
            }
        }
         return true;
    }

    void projection_loop()
    {
        while (!isStopping()) 
        {
            std::unique_lock<std::mutex> lk(m);
            signal.wait(lk, [this]{return projection_available == false;});
            // project the current state
            complexProjection(si_cad, camera_pose, warp_handler.state_current, proj_rgb);

            // get the ROI of the current state
            img_handler.set_projection_rois(proj_rgb, 20);
            // process the projection
            img_handler.setProcProj(proj_rgb);
            projection_available = true;
            proj_count++;
        }
    }

    void warp_loop()
    {
        bool updated = false;
        while (!isStopping()) 
        {
            if(projection_available) 
            {
                // set the current image
                img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);

                // warp the current projection based on the warp list
                //  and clear the list
                warp_handler.warp_by_history(warp_handler.projection.img_warp);

                // copy over the region of interest (needs to be thread safe)
                img_handler.set_obs_rois_from_projected();
                warp_handler.scale = img_handler.scale;

                updated = true;
                projection_available = false;
                signal.notify_one();
            }

            // perform warps
            if(updated)
                warp_handler.make_predictive_warps();

            // get the current EROS
            img_handler.setProcObs(eros_handler.eros.getSurface());
            warp_handler.proc_obs = img_handler.proc_obs;

            // perform the comparison
            warp_handler.score_predictive_warps();

            // update the state and update the warped image projection
            if (run) {
                // warp_handler.update_from_max();
                // warp_handler.update_all_possible();
                updated = warp_handler.update_heuristically();
                //updated = warp_handler.update_from_max();
                // state = warp_handler.state_current;
                //step = true;
            }
            warp_count++;
        }
        projection_available = false;
        signal.notify_one();
    }

    void sequential_loop()
    {
        double dataset_time = -1;
        bool updated = true;

        projectStateYawPitch(state);
        warp_handler.make_predictive_warps();
        

        while (!isStopping()) {
            updated = true;

            double dtic = Time::now();
            double dtoc_proj = Time::now();
            double dtoc_projproc = Time::now();

            if(updated) {

                projectStateYawPitch(state);
                dtoc_proj = Time::now();

                warp_handler.make_predictive_warps();
                dtoc_projproc = Time::now();


            }

            // if(updated) {

            //     //perform the projection
            //     si_cad->superimpose(q2aa(state), q2aa(camera_pose), proj_rgb);

            //     //extract RoIs
            //     img_handler.set_projection_rois(proj_rgb);

            //     //and copy them also for the observation
            //     img_handler.set_obs_rois_from_projected();
            //     warp_handler.scale = img_handler.scale;
            //     //warp_handler.extract_rois(proj_rgb);

            //     //make the projection template
            //     img_handler.setProcProj(proj_rgb);
            //     img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);
            //     dtoc_proj = Time::now();
            
            //     //make predictions
            //     warp_handler.make_predictive_warps();
            //     replaceyawpitch(img_handler.img_roi);
            //     dtoc_projproc = Time::now();
            // }

            //get the EROS
            dataset_time = eros_handler.tic;
            static cv::Mat scaled_eros = cv::Mat::zeros(img_size, CV_8U);
            cv::resize(eros_handler.eros.getSurface(), scaled_eros, scaled_eros.size());
            img_handler.setProcObs(scaled_eros);
            //warp_handler.proc_obs = img_handler.proc_obs;
            img_handler.proc_obs.copyTo(warp_handler.proc_obs);
            double dtoc_eros = Time::now();

            warp_handler.score_predictive_warps();
            if(run) {
                //updated = warp_handler.update_from_max();
                //updated = warp_handler.update_all_possible();
                updated = warp_handler.update_heuristically();
                state = warp_handler.state_current;
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

    void projectStateYawPitch(std::array<double, 7> object)
    {
        
        std::array<double, 7> state_temp;
        std::vector<std::array<double, 7> > objects;
        static std::vector<cv::Mat> images;
        if (images.empty()) {
            for (auto i = 0; i < 5; i++)
                images.push_back(cv::Mat::zeros(img_size, CV_8U));
        }

        objects.push_back(q2aa(object));
        
        double pitch = 1 * M_PI_2 *  1.0 / (img_handler.img_roi.height*0.5);
        pitch = 2.0 * M_PI / 180.0;
        state_temp = object;
        perform_rotation(state_temp, 0, pitch);
        objects.push_back(q2aa(state_temp));

        state_temp = object;
        perform_rotation(state_temp, 0, -pitch);
        objects.push_back(q2aa(state_temp));

        double yaw = 1 * M_PI_2 *  1.0 / (img_handler.img_roi.width*0.5);
        yaw = 2.0 * M_PI / 180.0;
        state_temp = object;
        perform_rotation(state_temp, 1, yaw);
        objects.push_back(q2aa(state_temp));

        state_temp = object;
        perform_rotation(state_temp, 1, -yaw);
        objects.push_back(q2aa(state_temp));

        si_cad->superimpose(q2aa(camera_pose), objects, img_handler.img_roi, images);

        img_handler.set_projection_rois(images[0]);
        img_handler.set_obs_rois_from_projected();
        warp_handler.scale = img_handler.scale;

        img_handler.setProcProj(images[0]);
        img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);
        proj_rgb = images[0];

        img_handler.setProcProj(images[1]);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::ap].img_warp);
        warp_handler.warps[warpManager::ap].delta = pitch;

        img_handler.setProcProj(images[2]);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::an].img_warp);
        warp_handler.warps[warpManager::an].delta = -pitch;

        img_handler.setProcProj(images[3]);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::bp].img_warp);
        warp_handler.warps[warpManager::bp].delta = yaw;

        img_handler.setProcProj(images[4]);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::bn].img_warp);
        warp_handler.warps[warpManager::bn].delta = -yaw;

    }

    void replaceyawpitch(cv::Rect roi)
    {
        static cv::Mat rgb = cv::Mat::zeros(img_size, CV_8UC1);
        rgb = 0;
        cv::Mat rgb_roi = rgb(roi);
        //get the state change for delta pitch (around x axis)

        double theta = M_PI_2 *  3.0 / (roi.height*0.5);
        auto state_temp = state;
        perform_rotation(state_temp, 0, theta);
        si_cad->superimpose(q2aa(state_temp), q2aa(camera_pose), rgb_roi, roi);
        //complexProjection(si_cad, camera_pose, state_temp, rgb);
        img_handler.setProcProj(rgb);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::ap].img_warp);
        warp_handler.warps[warpManager::ap].delta = theta;

        state_temp = state;
        perform_rotation(state_temp, 0, -theta);
        si_cad->superimpose(q2aa(state_temp), q2aa(camera_pose), rgb_roi, roi);
        img_handler.setProcProj(rgb);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::an].img_warp);
        warp_handler.warps[warpManager::an].delta = -theta;

        theta = M_PI_2 *  3.0 / (roi.width*0.5);
        state_temp = state;
        perform_rotation(state_temp, 1, theta);
        si_cad->superimpose(q2aa(state_temp), q2aa(camera_pose), rgb_roi, roi);
        img_handler.setProcProj(rgb);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::bp].img_warp);
        warp_handler.warps[warpManager::bp].delta = theta;

        state_temp = state;
        perform_rotation(state_temp, 1, -theta);
        si_cad->superimpose(q2aa(state_temp), q2aa(camera_pose), rgb_roi, roi);
        img_handler.setProcProj(rgb);
        img_handler.proc_proj.copyTo(warp_handler.warps[warpManager::bn].img_warp);
        warp_handler.warps[warpManager::bn].delta = -theta;

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
        yInfo() << "waiting for worker threads ... ";
        if(parallel_method) {
            warp_worker.join();
            proj_worker.join();
        } else {
            worker.join();
        }
        if(vid_writer.isOpened())
            vid_writer.release();
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
    rf.setDefaultConfigFile("/usr/local/src/EDOPT/configGELATIN.ini");
    rf.configure(argc, argv);
    
    return my_tracker.runModule(rf);
}
