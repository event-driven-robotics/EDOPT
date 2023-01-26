#include <yarp/os/all.h>
#include <chrono>
#include "projection.h"
#include <thread>
#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

using namespace yarp::os;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

cv::Mat make_visualisation(cv::Mat image0, cv::Mat image1) {
    cv::Mat rgb_img, temp;
    std::vector<cv::Mat> channels;
    channels.resize(3);
    double minv, maxv, norm_val;

    cv::minMaxLoc(image0, &minv, &maxv);
    norm_val = std::max(fabs(minv), fabs(maxv));
    //cv::normalize(image0, temp, 0.0, 255.0, cv::NORM_MINMAX);
    image0.convertTo(channels[0], CV_8U, 127.0/norm_val, 127);

    cv::minMaxLoc(image1, &minv, &maxv);
    norm_val = std::max(fabs(minv), fabs(maxv));
    //cv::normalize(image1, temp, 0.0, 255.0, cv::NORM_MINMAX);
    image1.convertTo(channels[1], CV_8U, 127.0/norm_val, 127);

    channels[2] = cv::Mat(image0.size(), CV_8U);
    channels[2].setTo(0);

    cv::merge(channels, rgb_img);

    return rgb_img;
}

void enlarge(const cv::Mat &img, cv::Rect &roi, int k)
{
    roi.x -= k; if(roi.x < 0) roi.x = 0;
    roi.y -= k; if(roi.y < 0) roi.y = 0;
    roi.width += 2 * k; 
    if(roi.x + roi.width > img.cols) 
        roi.width = img.cols - roi.x;
    roi.height += 2 * k;
    if(roi.y + roi.height > img.rows)
        roi.height = img.rows - roi.y;
    //roi = roi & cv::Rect(0, 0, img.cols, img.rows);
}

class luv_renderer {
public:
    int tab_entries{72};
    SICAD* si_cad;
    std::array<double, 7> camera_pose, initial_state;
    std::array<double, 6> cam_calib;
    cv::Size img_size;
    cv::Point img_centre;
    cv::Point img_f;
    cv::Mat temp_img;
    imageProcessing ip;
    double ang_res{1.0};
    //const static cv::Vec3b white({255, 255, 255})
    

    std::vector<std::vector<cv::Mat>> table;

    bool generate_templates(ResourceFinder &rf) 
    {

        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        if(!loadPose(rf, "camera_pose", camera_pose)) return false;
        if(!loadPose(rf, "object_pose", initial_state)) return false;
        yarp::os::Bottle &intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Could not load camera parameters";
            return false;
        }
        img_size = cv::Size(intrinsic_parameters.find("w").asInt32(),
                    intrinsic_parameters.find("h").asInt32());
        img_centre = cv::Point(intrinsic_parameters.find("cx").asInt32(),
                    intrinsic_parameters.find("cy").asInt32());
        img_f = cv::Point(intrinsic_parameters.find("fx").asInt32(),
                    intrinsic_parameters.find("fy").asInt32());
        temp_img = cv::Mat(img_size, CV_8U);

        if(!rf.check("template_file")) {
            yError() << "Error Generating Templates: no template file provided";
            return false;
        }
        
        cv::VideoWriter vid_writer;
        vid_writer.open(rf.find("template_file").asString(), cv::VideoWriter::fourcc('H','2','6','4'), 10, img_size, false);
        if(!vid_writer.isOpened()) {
            yError() << "Could not open video writer";
            return false;
        }
    
        ang_res = rf.check("angular_resolution", Value(1)).asInt();
        if(ang_res < 0 || ang_res == 7 || ang_res > 10) {
            yError() << "valid angular_resolution [1 2 3 4 5 6 8 9 10]";
            return false;
        }
        tab_entries = 360 / ang_res;

        

        yInfo() << "Angular resolution:" << ang_res << "degrees"; 


        yInfo() << "Generating Projections";

        std::array<double, 7> temp_state;
        std::array<double, 3> pyr = extract_pyr(initial_state);
        std::array<double, 3> rotations = {0};
        cv::Mat temp;
        cv::Mat_<float> temp32F;
        si_cad->superimpose(q2aa(initial_state), q2aa(camera_pose), temp);
        for(int a = 0; a < tab_entries; a++) {
            rotations[0] = (ang_res * a * M_PI / 180.0) - M_PI + pyr[0];
            for(int b = 0; b < tab_entries; b++) {
                rotations[1] = (ang_res * b * M_PI / 180.0) - M_PI + pyr[1];
                temp_state = initial_state;
                perform_rotation(temp_state, rotations);
                si_cad->superimpose(q2aa(temp_state), q2aa(camera_pose), temp);
                cvRect det_roi = cv::boundingRect(temp);
                ip.make_template(temp, temp32F);
                temp32F.convertTo(temp, CV_8U, 127, 127);
                vid_writer << temp;
                //cv::rectangle(temp, cv::Rect(0, temp.rows*0.95, temp.cols*a/tab_entries, temp.rows*0.05), cv::Vec3b(255, 255, 255), cv::FILLED);
                //cv::imshow("table", temp);
                //cv::waitKey(1);
            }
            std::cout << a*100/tab_entries << "%        \r";
            std::cout.flush();
        }
        std::cout << "Done        " << std::endl;
        
        //cv::destroyWindow("table");
        vid_writer.release();

        


        return true;


    }

    bool load_templates(ResourceFinder &rf) 
    {
        if(!rf.check("template_file")) {
            yError() << "Error Loading Templates: no template file provided";
            return false;
        }

        cv::VideoCapture input_video;
        if (!input_video.open(rf.find("template_file").asString())) {
            std::cerr << "Could not open video at: " << rf.find("template_file").asString() << std::endl;
            return false;
        }
        int total_frames = input_video.get(cv::CAP_PROP_FRAME_COUNT);
        int width = (int)input_video.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = (int)input_video.get(cv::CAP_PROP_FRAME_HEIGHT);
        int fps = (int)input_video.get(cv::CAP_PROP_FPS);

        tab_entries = sqrt(total_frames);
        if(tab_entries * tab_entries != total_frames) {
            yError() << "template file does not have correct frame count";
            return false;
        }
        ang_res = 360 / tab_entries;

        cv::Mat onemat, onemat32;

        double tic = Time::now();
        for(int i = 0; i < 1000; i++) {
            int k = ((double)rand() / RAND_MAX)*total_frames;
            input_video.set(cv::CAP_PROP_POS_FRAMES, k);
            input_video.read(onemat);
            onemat.convertTo(onemat32, CV_32F, 1.0/127.0, -1);
        }
        yInfo() << 1000*(Time::now() - tic) / 1000 << "ms per image";
        return false;
            
            


        yInfo() << "Loading Templates";
        cv::Mat frame, frame32F;
        table.resize(tab_entries);
        for(int a = 0; a < tab_entries; a++) {
            table[a].resize(tab_entries);
            for(int b = 0; b < tab_entries; b++) {
                input_video >> frame;
                //cv::imshow("loaded data", frame);
                //frame.convertTo(table[a][b], CV_32F, 1.0/127.0, -1);
                //frame.copyTo(table[a][b]);
                // frame.convertTo(
                // table[a][b] 
                //cv::waitKey(1);
            }
            std::cout << a*100/tab_entries << "%        \r";
            std::cout.flush();
        }
        std::cout << "Done        " << std::endl;
        
        return true;


    }

    bool configure(ResourceFinder &rf)
    {
        si_cad = createProjectorClass(rf);
        if(!si_cad)
            return false;

        if(!loadPose(rf, "camera_pose", camera_pose)) return false;
        if(!loadPose(rf, "object_pose", initial_state)) return false;
        yarp::os::Bottle &intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
        if (intrinsic_parameters.isNull()) {
            yError() << "Could not load camera parameters";
            return false;
        }
        img_size = cv::Size(intrinsic_parameters.find("w").asInt32(),
                    intrinsic_parameters.find("h").asInt32());
        img_centre = cv::Point(intrinsic_parameters.find("cx").asInt32(),
                    intrinsic_parameters.find("cy").asInt32());
        img_f = cv::Point(intrinsic_parameters.find("fx").asInt32(),
                    intrinsic_parameters.find("fy").asInt32());
        temp_img = cv::Mat(img_size, CV_8U);

        yInfo() << "Generating Projections";

        std::array<double, 7> temp_state;
        std::array<double, 3> pyr = extract_pyr(initial_state);
        std::array<double, 3> rotations = {0};
        
        cv::Mat temp;
        si_cad->superimpose(q2aa(initial_state), q2aa(camera_pose), temp);
        ang_res = 360 / tab_entries;
        for(int a = 0; a < tab_entries; a++) {
            rotations[1] = (ang_res * a * M_PI / 180.0) - M_PI + pyr[1];
            for(int b = 0; b < tab_entries; b++) {
                rotations[0] = (ang_res * b * M_PI / 180.0) - M_PI + pyr[0];
                temp_state = initial_state;
                perform_rotation(temp_state, rotations);
                si_cad->superimpose(q2aa(temp_state), q2aa(camera_pose), temp);
                ip.make_template(temp, table[b][a]);
                //cv::imshow("table", table[b][a]);
                //cv::waitKey(1);
            }
            yInfo() << (a*100) / tab_entries << "%";
        }
        //cv::destroyWindow("table");
        return true;
    }

    void render_gpu(std::array<double, 7> state, cv::Mat &img, cv::Rect roi)
    {
        cv::Mat temp = img(roi);
        si_cad->superimpose(q2aa(state), q2aa(camera_pose), temp, roi);
    }

    void render_luv(std::array<double, 7> state, cv::Mat &img, cv::Rect &roi)
    {
        //calculate alpha and beta
        //get the current state and rotate it by the viewing angle
        double xd = (state[0]-initial_state[0]);
        double zd = (state[2]-initial_state[2]);
        double yd = (state[1]-initial_state[1]);

        double pxd = xd * img_f.x / state[2];
        double pyd = yd * img_f.y / state[2];


        std::array<double, 3> pyr = extract_pyr(state);

        pyr[1] -= atan2(state[0], state[2]);


        //get the image with the correct rotation, then apply the 
        //required affine
        cv::Mat &entry = query_table(0, pyr[1]);
        if(entry.empty()) {
            yError() << "Empty table entry";
            return;
        }
        roi = cv::Rect(175, 86, 300, 300);

        //then affine transform for the other components
        img.setTo(0);
        if(!fast_affine(entry, img, roi, pxd, pyd, 0, 1.0))
            std::cout << "scale too big" << std::endl;

    }

    //assume input and output are the same size.
    bool fast_affine(cv::Mat &input, cv::Mat &output, cv::Rect roi, int dx, int dy, double roll, double scale)
    {
        //dy = -dy;

        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(roi.width/2, roi.height/2), roll, scale);

        //the output roi needs to be modified to include the scale.

        int x_offset = (roi.width - roi.width*scale)*0.5;
        int y_offset = (roi.height - roi.height*scale)*0.5;

        //first we need to rotate and scale in place
        cv::Rect output_roi(roi.x+dx+x_offset, roi.y+dy+y_offset, roi.width*scale, roi.height*scale);
        output_roi &= cv::Rect(0, 0, output.cols, output.rows);
        if(scale > 1.0)
            roi = cv::Rect(roi.x+x_offset, roi.y+y_offset, roi.width*scale, roi.height*scale);

        if(roi.x < 0 || roi.y < 0 || roi.x+roi.width > input.cols || roi.y + roi.height > input.rows)
            return false;
        //input(roi).copyTo(output(output_roi));
        
        cv::warpAffine(input(roi), output(output_roi), rot, output_roi.size());

        return true;

    }

    cv::Mat &query_table(double pitch, double yaw)
    {
        int a = (int)(((180.0 *    yaw / M_PI) + 180.0)/ang_res - 0.5)%tab_entries;
        int b = (int)(((180.0 *  pitch / M_PI) + 180.0)/ang_res - 0.5)%tab_entries;
        //yInfo() << pitch * 180.0/M_PI << " ->" << b;
        //yInfo() << yaw * 180.0/M_PI << " ->" << a; 
        return table[b][a];
    }

};

static int wait_time = 100;
static bool run_flag = true;
void show_image()
{
    cv::namedWindow("LUV vs GPU", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("LUV vs GPU", 1920, 10);
    cv::resizeWindow("LUV vs GPU", cv::Size(640, 480));
    
    // cv::namedWindow("GPU", cv::WINDOW_AUTOSIZE);
    // cv::moveWindow("GPU", 1920+800, 10);

    while(run_flag)
        if(cv::waitKey(wait_time) == 27);
            run_flag = false;
}



int main(int argc, char* argv[])
{

    std::thread second_thread;

    imageProcessing ip;

    std::array<double, 7> initial_state, camera_pose, state;
    luv_renderer luvr;

    ResourceFinder rf;
    rf.setDefaultConfigFile("/usr/local/src/object-track-6dof/configCAR.ini");
    rf.configure(argc, argv);

    yarp::os::Bottle& intrinsic_parameters = rf.findGroup("CAMERA_CALIBRATION");
    if (intrinsic_parameters.isNull()) {
        yError() << "Wrong .ini file or [CAMERA_CALIBRATION] not present. Deal breaker.";
        return false;
    }
    int w = intrinsic_parameters.find("w").asInt32();
    int h = intrinsic_parameters.find("h").asInt32();
    // warp_handler.cam[warpManager::cx] = intrinsic_parameters.find("cx").asFloat32()*render_scaler;
    // warp_handler.cam[warpManager::cy] = intrinsic_parameters.find("cy").asFloat32()*render_scaler;
    // warp_handler.cam[warpManager::fx] = intrinsic_parameters.find("fx").asFloat32()*render_scaler;
    // warp_handler.cam[warpManager::fy] = intrinsic_parameters.find("fy").asFloat32()*render_scaler;
    //img_size = cv::Size(warp_handler.cam[warpManager::x], warp_handler.cam[warpManager::y]);

    if(!loadPose(rf, "object_pose", initial_state)) return false;
    
    state = initial_state;

    //luvr.configure(rf);
    if(!luvr.load_templates(rf)) {
        if(!luvr.generate_templates(rf)) {
            yError() << "Could not load or generate templates";
            return false;
        }
        if(!luvr.load_templates(rf)) {
            yError() << "Could not load generated templates";
            return false;
        }
    }
    wait_time = 0;
    return true;
    //luvr.generate_templates(rf);
    // luvr.load_templates(rf);
    // return 0;

    second_thread = std::thread([]{show_image();});

    cv::Mat img_gpu(h, w, CV_8U); img_gpu.setTo(0);
    cv::Mat img_luv(h, w, CV_32F); img_luv.setTo(0.0);
    cv::Rect roi_gpu = cv::Rect(0, 0, w, h);
    cv::Rect roi_luv = cv::Rect(0, 0, w, h);
    cv::Mat temp(h, w, CV_32F);


    double gpu_time = 0.0, luv_time = 0.0, count = 0.0;
    //first render is problematic
    luvr.render_gpu(state, img_gpu, roi_gpu);
    int cs = 1;

    double increment = 0.1;
    for(double x = -50.0; x < 50; x+=increment) {
        for(double y = 0; y < increment; y+=increment) {
        state[0] = x;
        //state[1] = y;

        auto t1 = high_resolution_clock::now();
        for(int i = 0; i < cs; i++)
            luvr.render_luv(state, img_luv, roi_luv);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        luv_time += ms_double.count();

        //cv::rectangle(img_luv, roi_luv, cv::Vec3b(255, 255, 255));


        auto t3 = high_resolution_clock::now();
        for(int i = 0; i < cs; i++)
            luvr.render_gpu(state, img_gpu, roi_gpu);
        roi_gpu = cv::boundingRect(img_gpu);
        enlarge(img_gpu, roi_gpu, 10);
        
        ip.make_template(img_gpu, temp);
        
        auto t4 = high_resolution_clock::now();
        duration<double, std::milli> ms2_double = t4 - t3;
        gpu_time += ms2_double.count();

        //cv::rectangle(img_gpu, roi_gpu, cv::Vec3b(255, 255, 255));
        cv::Mat merged = make_visualisation(img_luv, temp);
        cv::imshow("LUV vs GPU", merged);

        //Time::delay(0.001);
        count += cs;
        //perform_rotation(state, 0, increment);
        // if(cv::waitKey(1) == 27)
        //     break;

        }
        if(run_flag == false)
            break;
    }
    std::cout <<  gpu_time / count <<" <-GPU | LUV-> " <<  luv_time / count << std::endl;
    gpu_time = 0.0, luv_time = 0.0, count = 0.0;
    wait_time = 3000;


    run_flag = false;
    second_thread.join();

    return 0;

}
