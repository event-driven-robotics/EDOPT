#pragma once

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>
#include <yarp/os/all.h>
#include <opencv2/opencv.hpp>

SICAD* createProjectorClass(yarp::os::ResourceFinder &config)
{


    if(!config.check("object_path")) {
        yError() << "Did not find object path";
        return nullptr;
    }

    yarp::os::Bottle &intrinsic_parameters = config.findGroup("CAMERA_CALIBRATION");
    if(intrinsic_parameters.isNull()) {
        yError() << "Could not load camera parameters";
        return nullptr;
    }
    std::string object_path = config.find("object_path").asString();
    SICAD::ModelPathContainer obj;
    obj.emplace("model", object_path);

        yInfo() << "Creating SICAD class with object: " << object_path
                << "and parameters" << intrinsic_parameters.toString();


    return new SICAD(obj,
                     intrinsic_parameters.find("w").asInt(),
                     intrinsic_parameters.find("h").asInt(),
                     intrinsic_parameters.find("fx").asDouble(),
                     intrinsic_parameters.find("fy").asDouble(),
                     intrinsic_parameters.find("cx").asDouble(),
                     intrinsic_parameters.find("cy").asDouble());

}

bool cameraBasedProjection(SICAD *si_cad, Superimpose::ModelPose pose, cv::Mat &image) {

    Superimpose::ModelPoseContainer objpose_map;

    std::vector<double> cam_pos(pose.begin(), pose.begin()+3);
    std::vector<double> cam_rot(pose.begin()+3, pose.end());
    static std::vector<double> origin = {0, 0, 0, 1, 0, 0, 0};
    objpose_map.emplace("model", origin);

    return si_cad->superimpose(objpose_map, cam_pos.data(), cam_rot.data(), image);

}


bool simpleProjection(SICAD *si_cad, Superimpose::ModelPose pose, cv::Mat &image) {

    Superimpose::ModelPoseContainer objpose_map;
    objpose_map.emplace("model", pose);

    std::vector<double> cam_pos = {0, 0, 0};
    std::vector<double> cam_rot = {1, 0, 0, 0}; //{0 0 0 0} is invalid

    return si_cad->superimpose(objpose_map, cam_pos.data(), cam_rot.data(), image);

}

bool simpleProjection(SICAD *si_cad, std::vector<SICAD::ModelPoseContainer>& objpos_multimap,
                      cv::Mat &image) {

    std::vector<double> cam_pos = {0, 0, 0};
    std::vector<double> cam_rot = {1, 0, 0, 0}; //{0 0 0 0} is invalid

    return si_cad->superimpose(objpos_multimap, cam_pos.data(), cam_rot.data(), image);

}

bool loadObjectPose(yarp::os::ResourceFinder &config, Superimpose::ModelPose &obj_pose,
                    std::string object_pose_name)
{
    yarp::os::Bottle *loaded_pose = config.find(object_pose_name).asList();
    if(!loaded_pose)
        return false;

    obj_pose.clear();
    for(size_t i = 0; i < loaded_pose->size(); i++)
        obj_pose.push_back(loaded_pose->get(i).asDouble());

    return true;

}


void normalise_quaternion(std::vector<double> &state)
{

    double normval = 1.0 / sqrt(state[3]*state[3] + state[4]*state[4] +
                                state[5]*state[5] + state[6]*state[6]);
    state[3] *= normval;
    state[4] *= normval;
    state[5] *= normval;
    state[6] *= normval;
}

Superimpose::ModelPose quaternion_to_axisangle(const std::vector<double> &state)
{
    Superimpose::ModelPose pose;
    pose.resize(7);

    pose[0] = state[0]; //x
    pose[1] = state[1]; //y
    pose[2] = state[2]; //z

    //if acos return -nan it means the quaternion wasn't normalised !
    pose[6] = 2 * acos(state[3]); //state[3] is w, pose[6] = angle
    double scaler =  sqrt(1 - state[3]*state[3]);
    if(scaler < 0.001) { //angle is close to 0 so it is insignificant (but don't divide by 0)
        pose[6] = 0.0; //angle
        pose[3] = 1.0; //ax
        pose[4] = 0.0; //ay
        pose[5] = 0.0; //az
    } else {
        scaler = 1.0 / scaler;
        pose[3] = state[4] * scaler;
        pose[4] = state[5] * scaler;
        pose[5] = state[6] * scaler;
    }

    return pose;
}

Superimpose::ModelPose euler_to_axisangle(const std::vector<double> &state)
{
    static constexpr double halfdeg2rad = 0.5 * 2.0 * M_PI / 180.0;
    Superimpose::ModelPose pose;
    pose.resize(7);

    double c1 = cos(state[3]*halfdeg2rad);
    double c2 = cos(state[4]*halfdeg2rad);
    double c3 = cos(state[5]*halfdeg2rad);
    double s1 = sin(state[3]*halfdeg2rad);
    double s2 = sin(state[4]*halfdeg2rad);
    double s3 = sin(state[5]*halfdeg2rad);

    pose[0] = state[0];
    pose[1] = state[1];
    pose[2] = state[2];


    pose[3] = s1*s2*c3 + c1*c2*s3;
    pose[4] = s1*c2*c3 + c1*s2*s3;
    pose[5] = c1*s2*c3 - s1*c2*s3;

    double norm = pose[3]*pose[3] + pose[4]*pose[4] + pose[5]*pose[5];
    if(norm < 0.001) {
        pose[3] = 1.0;
        pose[4] = pose[5] = pose[6] = 0.0;
    } else {
        norm = sqrt(norm);
        pose[3] /= norm;
        pose[4] /= norm;
        pose[5] /= norm;
        pose[6] = 2.0 * acos(c1*c2*c3 - s1*s2*s3);
    }



    return pose;
}


cv::Mat preprocimg(cv::Mat projected)
{
    static cv::Mat canny_img;
    cv::Canny(projected, canny_img, 40, 40*3, 3);
    cv::dilate(canny_img, canny_img, cv::Mat());
    //cv::GaussianBlur(canny_img, canny_img, cv::Size(5, 5), 0);
    return canny_img;
}

cv::Mat mhat(cv::Mat projected)
{
    static cv::Mat canny_img, f, pos_hat, neg_hat;
    cv::Canny(projected, canny_img, 40, 40*3, 3);
    
    //cv::dilate(canny_img, canny_img, cv::Mat());
    canny_img.convertTo(f, CV_32F);
    cv::GaussianBlur(f, pos_hat, cv::Size(21, 21), 0);
    cv::GaussianBlur(f, neg_hat, cv::Size(41, 41), 0);

    cv::Mat grey = cv::Mat::zeros(canny_img.size(), CV_32F);
    grey -= neg_hat;
    grey += pos_hat;

    double minval, maxval;
    cv::minMaxLoc(grey, &minval, &maxval);
    maxval = 2*std::max(fabs(minval), fabs(maxval));
    grey /= maxval;

    return grey;
}

cv::Mat erosproc(cv::Mat eros_img)
{
    //get the summed image with kernel 7x7

    //get the difference between 

    return cv::Mat();

}