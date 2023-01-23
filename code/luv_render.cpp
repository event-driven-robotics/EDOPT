#include <yarp/os/all.h>
#include <chrono>
#include "projection.h"
#include <thread>
#include "image_processing.h"

using namespace yarp::os;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

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
    SICAD* si_cad;
    std::array<double, 7> camera_pose, initial_state;
    std::array<double, 6> cam_calib;
    cv::Size img_size;
    cv::Point img_centre;
    cv::Point img_f;
    cv::Mat temp_img;
    imageProcessing ip;
    

    std::array<cv::Mat, 360> table;

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

        si_cad->superimpose(q2aa(initial_state), q2aa(camera_pose), table[0]);
        double base_angle = extract_yaw(initial_state);
        double increment = 2 * M_PI / 360;  
        cv::Mat temp;
        for(int a = 0; a < 360; a++) {
            double angle = (a * M_PI / 180.0) - M_PI + base_angle;
            //table[a] = cv::Mat(img_size, CV_8U);
            std::array<double, 7> temp_state = initial_state;
            perform_rotation(temp_state, 1, -angle);
            si_cad->superimpose(q2aa(temp_state), q2aa(camera_pose), temp);
            ip.make_template(temp, table[a]);
             cv::imshow("table", table[a]);
             cv::waitKey(1);
        }
        cv::destroyWindow("table");


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
        //perform_rotation(state, 0, atan2(xd, zd));
        double alpha = extract_yaw(state);

        //calculate the required alpha and beta
        cv::Mat &entry = query_table(alpha, 0);
        //roi = cv::boundingRect(entry);
        //std::cout << roi << std::endl;
        //enlarge(entry, roi, 10);
        roi = cv::Rect(175, 86, 300, 300);

        //then affine transform for the other components
        
        double pxd = xd * img_f.x / state[2];

        //cv::Mat M = (cv::Mat_<double>(2,3) << 1.0, 0, pxd, 0, 1.0, 0);
        //cv::warpAffine(entry, img, M, img.size());
        img.setTo(0);
        if(!fast_affine(entry, img, roi, pxd, -40, 45, 1.1))
            std::cout << "scale too big" << std::endl;
        //entry.copyTo(img);

    }

    //assume input and output are the same size.
    bool fast_affine(cv::Mat &input, cv::Mat &output, cv::Rect roi, int dx, int dy, double angle, double scale)
    {
        dy = -dy;

        cv::Mat rot = cv::getRotationMatrix2D(cv::Point(roi.width/2, roi.height/2), angle, scale);

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

    cv::Mat &query_table(double alpha, double beta)
    {
        int a = (int)((180.0 * alpha / M_PI) + 180)%360;
        //std::cout << a << std::endl;
        //si_cad->superimpose(q2aa(initial_state), q2aa(camera_pose), temp_img);

        return table[a];
    }

};

static int wait_time = 100;
static bool run_flag = true;
void show_image()
{
    cv::namedWindow("LUV", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("LUV", 0, 10);
    
    cv::namedWindow("GPU", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("GPU", 0+800, 10);

    while(run_flag)
        cv::waitKey(wait_time);
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



    second_thread = std::thread([]{show_image();});

    luvr.configure(rf);
    cv::Mat img_gpu(h, w, CV_8U); img_gpu.setTo(0);
    cv::Mat img_luv(h, w, CV_32F); img_luv.setTo(0.0);
    cv::Rect roi_gpu = cv::Rect(0, 0, w, h);
    cv::Rect roi_luv = cv::Rect(0, 0, w, h);
    cv::Mat temp(h, w, CV_32F);

    double gpu_time = 0.0, luv_time = 0.0, count = 0.0;
    //first render is problematic
    luvr.render_gpu(state, img_gpu, roi_gpu);
    int cs = 1;

    double increment = 0.5;
    for(double x = -50.0; x < 50; x+=increment) {
        for(double y = -20.0; y < 20; y+=increment) {
        state[0] = x;
        state[1] = y;

        auto t1 = high_resolution_clock::now();
        for(int i = 0; i < cs; i++)
            luvr.render_luv(state, img_luv, roi_luv);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        luv_time += ms_double.count();

        cv::rectangle(img_luv, roi_luv, cv::Vec3b(255, 255, 255));
        cv::imshow("LUV", img_luv);

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
        cv::imshow("GPU", temp);

        //Time::delay(0.001);
        count += cs;
        //perform_rotation(state, 0, increment);
        // if(cv::waitKey(1) == 27)
        //     break;
        }
    }
    std::cout <<  gpu_time / count <<" <-GPU | LUV-> " <<  luv_time / count << std::endl;
    gpu_time = 0.0, luv_time = 0.0, count = 0.0;
    wait_time = 3000;


    run_flag = false;
    second_thread.join();

    return 0;

}
