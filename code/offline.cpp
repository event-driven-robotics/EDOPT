

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

    std::string datapath;
    imageProcessing img_handler;
    warpManager warp_handler;

    ev::offlineLoader<ev::AE> eloader;
    ev::SCARF scarf;

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

    std::ofstream outFile;
    double updateT = 0.001; 

public:
    struct DataPoint {
        double timestamp;
        double x, y, z;
        double qx, qy, qz, qw;
    };

    bool configure(yarp::os::ResourceFinder& rf) override
    {
        setName(rf.check("name", Value("/ekom")).asString().c_str());
        double bias_sens = rf.check("s", Value(0.6)).asFloat64();
        double cam_filter = rf.check("f", Value(0.1)).asFloat64();

        datapath = rf.check("data", Value("/data/mustard_bottle_translation_xyz_roll_edopt_test1/leftdvs/data.log")).asString();

        proc_size = rf.check("proc_size", Value(100)).asInt32();
        canny_thresh = rf.check("canny_thresh", Value(40)).asInt32();
        canny_scale = rf.check("canny_scale", Value(3)).asFloat64();
        block_size = rf.check("block_size", Value(14)).asInt32();
        alpha = rf.check("alpha", Value(2.0)).asFloat64();
        c_factor = rf.check("c", Value(0.3)).asFloat64();
        period = rf.check("period", Value(0.1)).asFloat64();
        dp2 = rf.check("dp2") && rf.check("dp2", Value(true)).asBool(); //default false
        dp2 = true;
        //run = rf.check("run") && !rf.find("run").asBool() ? false : true; //default true
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

        double seconds = 60;
        double data_timelength = 0;

        if (!eloader.load(datapath, seconds)) {
            yError() << "Could not open data file" << datapath;
            return false;
        }
        else {
            yInfo() << eloader.getinfo();
            data_timelength = eloader.getLength();
            yInfo() << "Data time length [s]: " << data_timelength;
        }

        eloader.synchroniseRealtimeRead(0.0);

        // Initialize SCARF
        scarf.initialise(img_size, block_size, alpha, c_factor);

        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        int blur = proc_size / 20;
        double dp = 1;//+rescale_size / 100;
        warp_handler.initialise(proc_size, dp2);
        warp_handler.create_Ms(dp);
        warp_handler.set_current(state);
        img_handler.initialise(proc_size, blur, canny_thresh, canny_scale);

        proj_rgb = cv::Mat::zeros(img_size, CV_8UC1);
        proj_32f = cv::Mat::zeros(img_size, CV_32F);

        cv::namedWindow("SCARF", cv::WINDOW_NORMAL);
        cv::resizeWindow("SCARF", img_size);
        cv::moveWindow("SCARF", 1920, 100);

        // cv::namedWindow("Translations", cv::WINDOW_AUTOSIZE);
        // cv::resizeWindow("Translations", img_size);
        // cv::moveWindow("Translations", 0, 540);

        // cv::namedWindow("Rotations", cv::WINDOW_AUTOSIZE);
        // cv::resizeWindow("Rotations", img_size);
        // cv::moveWindow("Rotations", 700, 540);

        //quaternion_test();
        //quaternion_test_camera();

        worker = std::thread([this]{offline_loader();});


        // if(parallel_method) {
        //     proj_worker = std::thread([this]{projection_loop();});
        //     warp_worker = std::thread([this]{warp_loop();});
        // } else {
        //     worker = std::thread([this]{sequential_loop();});
        // }


        outFile.open("/data/object_state_t_"+std::to_string(updateT)+"_dp_"+std::to_string(dp)+".csv");
        if (!outFile.is_open()) {
            yError() << "Could not open output file" << "object_state.csv";
            return false;
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
        yInfo() << "SCARF:" << block_size << "," << alpha << "," << c_factor << "(--block_size <int> --alpha <float> --c <float>)";
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
        return true;
    }

    cv::Mat addScoreToImage(const cv::Mat& image, double score, std::string warp_name) {
        cv::Mat imgWithText = image.clone();
        std::string text;
        if (score != 0)
            text = warp_name + ": " + std::to_string(score);
        else
            text = warp_name;

        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Scalar textColor = 255; 
        cv::Point textPosition(10, 20); // Top-left corner
    
        cv::putText(imgWithText, text, textPosition, fontFace, fontScale, textColor, thickness);
        return imgWithText;
    }

    void offline_loader(){

        double seconds = 60;
        cv::Mat vis, proj_vis, scarf_vis;
        int row_img = 0;
        double timer = 0; 
        while(eloader.incrementReadTill(timer)){

            for (ev::offlineLoader<ev::AE>::iterator v = eloader.begin(); v != eloader.end(); v++) {    
                scarf.update(v->x, v->y, v->p);
            }
        
            // project the current state
            complexProjection(si_cad, camera_pose, warp_handler.state_current, proj_rgb);
            projectStateYawPitch(warp_handler.state_current);
            // get the ROI of the current state
            img_handler.set_projection_rois(proj_rgb, 20);
            // process the projection
            img_handler.setProcProj(proj_rgb);

            // set the current image
            img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);

            // warp the current projection based on the warp list
            //  and clear the list
            warp_handler.warp_by_history(warp_handler.projection.img_warp);

            // copy over the region of interest (needs to be thread safe)
            img_handler.set_obs_rois_from_projected();
            warp_handler.scale = img_handler.scale;

            // // project the current state
            // complexProjection(si_cad, camera_pose, warp_handler.state_current, proj_rgb);
            // projectStateYawPitch(warp_handler.state_current);
            // // get the ROI of the current state
            // img_handler.set_projection_rois(proj_rgb, 20);
            // // process the projection
            // img_handler.setProcProj(proj_rgb);

            warp_handler.make_predictive_warps();

            // get the current EROS
            img_handler.setProcObs(scarf.getSurface());
            warp_handler.proc_obs = img_handler.proc_obs;


            // perform the comparison
            warp_handler.score_predictive_warps();

            // update the state and update the warped image projection

            // warp_handler.update_from_max();
            // warp_handler.update_all_possible();
            warp_handler.update_heuristically();
            //updated = warp_handler.update_from_max();
            // state = warp_handler.state_current;
            //step = true;

            cv::Mat img32 = scarf.getSurface();
            cv::Mat img8U;
            img32.convertTo(img8U, CV_8U, 255);
            cv::cvtColor(img8U, scarf_vis, cv::COLOR_GRAY2BGR);
            cv::resize(scarf_vis, scarf_vis, img_size);
            cv::cvtColor(proj_rgb, proj_vis, cv::COLOR_GRAY2BGR);
            vis = proj_vis + scarf_vis;
            
            std::vector<cv::Mat> rowImages;
            cv::Mat finalImage;
            enum warp_name {
                xp, yp, zp, ap, bp, cp, xn, yn, zn, an, bn, cn,
                xp2, yp2, zp2, ap2, bp2, cp2, xn2, yn2, zn2, an2, bn2, cn2
            };
            std::vector<std::string> proj_name{"xp","yp","zp", "ap", "bp", "cp", "xn", "yn", "zn", "an", "bn", "cn"};
            // Define the order of pairs for stacking (12 rows, 2 columns)
            std::vector<std::pair<warp_name, warp_name>> warp_pairs = {
                {xp, xn}, {yp, yn}, {zp, zn}, {cp, cn},
                {ap, an}, {bp, bn}, {xp2, yp2}, {zp2, ap2},
                {bp2, cp2}, {xn2, yn2}, {zn2, an2}, {bn2, cn2}
            };
            cv::Mat firstrow;
            cv::Mat img1 = addScoreToImage(warp_handler.proc_obs, 0, "scarf");
            cv::Mat img2 = addScoreToImage(warp_handler.projection.img_warp, warp_handler.projection.score, "no motion");      

            cv::Mat proc_obs_8U;
            warp_handler.proc_obs.convertTo(proc_obs_8U, CV_8U, 255); 

            double max_score = warp_handler.projection.score; 
            int index_max = 12; 

            cv::hconcat(img1, img2, firstrow);
            rowImages.push_back(firstrow);

            for (const auto& pair : warp_pairs) {
                warp_name idx1 = pair.first;
                warp_name idx2 = pair.second;

                if (idx2 >= 12) {
                    // std::cerr << "Index out of bounds!" << std::endl;
                    continue;
                }

                if (warp_handler.warps[idx1].score > max_score){
                    max_score = warp_handler.warps[idx1].score;
                    index_max = idx1; 
                }
                if (warp_handler.warps[idx2].score > max_score){
                    max_score = warp_handler.warps[idx2].score;
                    index_max = idx2; 
                }

                cv::Mat img1 = addScoreToImage(warp_handler.warps[idx1].img_warp, warp_handler.warps[idx1].score, proj_name[idx1]);
                cv::Mat img2 = addScoreToImage(warp_handler.warps[idx2].img_warp, warp_handler.warps[idx2].score, proj_name[idx2]);        

                cv::Mat img1Color;
                cv::cvtColor(img1, img1Color, cv::COLOR_GRAY2BGR);

                cv::Mat img2Color;
                cv::cvtColor(img2, img2Color, cv::COLOR_GRAY2BGR);

                cv::Mat mask = proc_obs_8U > 0;  // Create a binary mask where proc_obs_8U > 0
                img1Color.setTo(cv::Scalar(0, 255, 0), mask);  // Set those pixels to green

                imshow("overlay", img1Color);
                cv::Mat hstacked;
                cv::hconcat(img1, img2, hstacked);

                rowImages.push_back(hstacked);

                row_img++; 
            }
            cv::vconcat(rowImages, finalImage);

            if (index_max != 12)
                std::cout<<"Winning axis = "<<proj_name[index_max]<<", score = "<<max_score<<std::endl; 
            else
                std::cout<<"Winning axis = no motion, score = "<<max_score<<std::endl; 

            rowImages.clear();
            row_img = 0; 

            cv::imshow("SCARF", vis);
            cv::imshow("final img", finalImage);

            outFile << timer << "," << warp_handler.state_current[0] << "," << warp_handler.state_current[1] << "," << warp_handler.state_current[2] << "," << warp_handler.state_current[3] << "," << warp_handler.state_current[4] << "," << warp_handler.state_current[5] << ","<< warp_handler.state_current[6]<< std::endl; 

            cv::waitKey(1);

            timer += updateT; 
            
        }

    }

    // void projection_loop()
    // {
    //     while (!isStopping()) 
    //     {
    //         std::unique_lock<std::mutex> lk(m);
    //         signal.wait(lk, [this]{return projection_available == false;});
    //         // project the current state
    //         complexProjection(si_cad, camera_pose, warp_handler.state_current, proj_rgb);

    //         // get the ROI of the current state
    //         img_handler.set_projection_rois(proj_rgb, 20);
    //         // process the projection
    //         img_handler.setProcProj(proj_rgb);
    //         projection_available = true;
    //         proj_count++;
    //     }
    // }

    // void warp_loop()
    // {
    //     bool updated = false;
    //     while (!isStopping()) 
    //     {
    //         if(projection_available) 
    //         {
    //             // set the current image
    //             img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);

    //             // warp the current projection based on the warp list
    //             //  and clear the list
    //             warp_handler.warp_by_history(warp_handler.projection.img_warp);

    //             // copy over the region of interest (needs to be thread safe)
    //             img_handler.set_obs_rois_from_projected();
    //             warp_handler.scale = img_handler.scale;

    //             updated = true;
    //             projection_available = false;
    //             signal.notify_one();
    //         }

    //         // perform warps
    //         if(updated)
    //             warp_handler.make_predictive_warps();

    //         // get the current EROS
    //         img_handler.setProcObs(scarf.getSurface());
    //         warp_handler.proc_obs = img_handler.proc_obs;

    //         // perform the comparison
    //         warp_handler.score_predictive_warps();

    //         // update the state and update the warped image projection
    //         if (run) {
    //             // warp_handler.update_from_max();
    //             // warp_handler.update_all_possible();
    //             updated = warp_handler.update_heuristically();
    //             //updated = warp_handler.update_from_max();
    //             // state = warp_handler.state_current;
    //             //step = true;
    //         }
    //         warp_count++;
    //     }
    //     projection_available = false;
    //     signal.notify_one();
    // }

    // void sequential_loop()
    // {
    //     double dataset_time = -1;
    //     bool updated = true;

    //     projectStateYawPitch(state);
    //     warp_handler.make_predictive_warps();
        

    //     while (!isStopping()) {
    //         updated = true;

    //         double dtic = Time::now();
    //         double dtoc_proj = Time::now();
    //         double dtoc_projproc = Time::now();

    //         if(updated) {

    //             projectStateYawPitch(state);
    //             dtoc_proj = Time::now();

    //             warp_handler.make_predictive_warps();
    //             dtoc_projproc = Time::now();


    //         }

    //         // if(updated) {

    //         //     //perform the projection
    //         //     si_cad->superimpose(q2aa(state), q2aa(camera_pose), proj_rgb);

    //         //     //extract RoIs
    //         //     img_handler.set_projection_rois(proj_rgb);

    //         //     //and copy them also for the observation
    //         //     img_handler.set_obs_rois_from_projected();
    //         //     warp_handler.scale = img_handler.scale;
    //         //     //warp_handler.extract_rois(proj_rgb);

    //         //     //make the projection template
    //         //     img_handler.setProcProj(proj_rgb);
    //         //     img_handler.proc_proj.copyTo(warp_handler.projection.img_warp);
    //         //     dtoc_proj = Time::now();
            
    //         //     //make predictions
    //         //     warp_handler.make_predictive_warps();
    //         //     replaceyawpitch(img_handler.img_roi);
    //         //     dtoc_projproc = Time::now();
    //         // }

    //         //get the EROS
    //         dataset_time = eros_handler.tic;
    //         static cv::Mat scaled_eros = cv::Mat::zeros(img_size, CV_8U);
    //         cv::resize(eros_handler.eros.getSurface(), scaled_eros, scaled_eros.size());
    //         img_handler.setProcObs(scaled_eros);
    //         //warp_handler.proc_obs = img_handler.proc_obs;
    //         img_handler.proc_obs.copyTo(warp_handler.proc_obs);
    //         double dtoc_eros = Time::now();

    //         warp_handler.score_predictive_warps();
    //         if(run) {
    //             //updated = warp_handler.update_from_max();
    //             //updated = warp_handler.update_all_possible();
    //             updated = warp_handler.update_heuristically();
    //             state = warp_handler.state_current;
    //         }
            
    //         double dtoc_warp = Time::now();
    //         //yInfo() << "update:"<< dtoc_eros - dtoc_warp;

    //         this->toc_eros += ((dtoc_eros - dtoc_projproc) * 1e6);
    //         this->toc_proj += ((dtoc_proj - dtic) * 1e6);
    //         this->toc_projproc += ((dtoc_projproc - dtoc_proj) * 1e6);
    //         this->toc_warp += ((dtoc_warp - dtoc_eros) * 1e6);
    //         this->toc_count++;
    //         if (fs.is_open() && dataset_time > 0) {
    //             data_to_save.push_back({dataset_time, state[0], state[1], state[2], state[3], state[4], state[5], state[6]});
    //         }
    //     }
    // }

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
        // eros_handler.stop();
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

        outFile.close(); 

        yInfo() << "close function finished";
        return true;
    }

};

int main(int argc, char* argv[])
{
    tracker my_tracker;
    ResourceFinder rf;
    rf.setDefaultConfigFile("/usr/local/src/EDOPT/configMUSTARD.ini");
    rf.configure(argc, argv);
    
    return my_tracker.runModule(rf);
}
