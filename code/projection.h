#pragma once

#include <SuperimposeMesh/SICAD.h>
#include <opencv2/opencv.hpp>
#include <yarp/os/all.h>

std::array<double, 7> q2aa(const std::array<double, 7> &state)
{
    std::array<double, 7> aa{0};

    aa[0] = state[0]; //x
    aa[1] = state[1]; //y
    aa[2] = state[2]; //z

    //if acos return -nan it means the quaternion wasn't normalised !
    aa[6] = 2 * acos(state[6]); //state[6] is w, pose[6] = angle
    double scaler =  sqrt(1 - state[6]*state[6]);
    if(scaler < 0.001) { //angle is close to 0 so it is insignificant (but don't divide by 0)
        aa[3] = 1.0; //ax
    } else {
        scaler = 1.0 / scaler;
        aa[3] = state[3] * scaler;
        aa[4] = state[4] * scaler;
        aa[5] = state[5] * scaler;
    }

    return aa;
}

Superimpose::ModelPose quaternion_to_axisangle(const std::array<double, 7> &state)
{
    Superimpose::ModelPose pose;
    pose.resize(7);

    pose[0] = state[0]; //x
    pose[1] = state[1]; //y
    pose[2] = state[2]; //z

    //if acos return -nan it means the quaternion wasn't normalised !
    pose[6] = 2 * acos(state[6]); //state[6] is w, pose[6] = angle
    double scaler =  sqrt(1 - state[6]*state[6]);
    if(scaler < 0.001) { //angle is close to 0 so it is insignificant (but don't divide by 0)
        pose[6] = 0.0; //angle
        pose[3] = 1.0; //ax
        pose[4] = 0.0; //ay
        pose[5] = 0.0; //az
    } else {
        scaler = 1.0 / scaler;
        pose[3] = state[3] * scaler;
        pose[4] = state[4] * scaler;
        pose[5] = state[5] * scaler;
    }

    return pose;
}


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
    
    double render_scaler = config.check("render_scaler", Value(1.0)).asFloat64();

    return new SICAD(obj,
                     intrinsic_parameters.find("w").asInt32()*render_scaler,
                     intrinsic_parameters.find("h").asInt32()*render_scaler,
                     intrinsic_parameters.find("fx").asFloat32()*render_scaler,
                     intrinsic_parameters.find("fy").asFloat32()*render_scaler,
                     intrinsic_parameters.find("cx").asFloat32()*render_scaler,
                     intrinsic_parameters.find("cy").asFloat32()*render_scaler);

}

bool complexProjection(SICAD *si_cad, const std::array<double, 7> &camera, const std::array<double, 7> &object, cv::Mat &image) {

    return si_cad->superimpose(q2aa(object), q2aa(camera), image);
    // Superimpose::ModelPoseContainer objpose_map;

    // Superimpose::ModelPose op = quaternion_to_axisangle(object);
    // Superimpose::ModelPose cp = quaternion_to_axisangle(camera);
    // objpose_map.emplace("model", op);

    // return si_cad->superimpose(objpose_map, &(cp[0]), &(cp[3]), image);

}

bool cameraBasedProjection(SICAD *si_cad, const std::array<double, 7> &camera, cv::Mat &image) 
{
    static std::array<double, 7> objectatorigin = {0, 0, 0, 1, 0, 0, 0};
    return complexProjection(si_cad, camera, objectatorigin, image);
}


bool simpleProjection(SICAD *si_cad, const std::array<double, 7> &object, cv::Mat &image) 
{
    static std::array<double, 7> cameraatorigin = {0, 0, 0, 1, 0, 0, 0};
    return complexProjection(si_cad, cameraatorigin, object, image);
}

bool loadPose(yarp::os::ResourceFinder &config, const std::string pose_name, std::array<double, 7> &pose)
{
    yarp::os::Bottle *loaded_pose = config.find(pose_name).asList();
    if(!loaded_pose) {
        yError() << "Could not find pose name";
        return false;
    }

    yInfo() << pose_name << loaded_pose->toString();

    if(loaded_pose->size() != pose.size()) {
        yError() << "Pose incorrect size: " << loaded_pose->size();
        return false;
    }

    for(size_t i = 0; i < pose.size(); i++)
        pose[i] = loaded_pose->get(i).asFloat32();

    return true;

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

void normalise_quaternion(std::array<double, 4> &q)
{

    double normval = 1.0 / sqrt(q[0]*q[0] + q[1]*q[1] +
                                q[2]*q[2] + q[3]*q[3]);
    q[0] *= normval;
    q[1] *= normval;
    q[2] *= normval;
    q[3] *= normval;
}

std::array<double, 4> quaternion_rotation(const std::array<double, 4> &q1, const std::array<double, 4> &q2)
{
    std::array<double, 4> q3;
    q3[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2];
    q3[0] = q1[3]*q2[0] + q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1];
    q3[1] = q1[3]*q2[1] + q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0];
    q3[2] = q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3];
    normalise_quaternion(q3);
    return q3;
} 

std::array<double, 4> create_quaternion(int axis, double radians)
{
    std::array<double, 4> q{0};
    axis %= 3;
    radians *= 0.5;
    q[axis] = sin(radians);
    q[3] = cos(radians);
    return q;
}


void perform_rotation(std::array<double, 7> &state, int axis, double radians)
{
    axis %= 3;
    std::array<double, 4> rq = create_quaternion(axis, radians);
    std::array<double, 4> q;
    q[0] = state[3];
    q[1] = state[4];
    q[2] = state[5];
    q[3] = state[6];
    std::array<double, 4> fq = quaternion_rotation(q, rq);
    state[3] = fq[0];
    state[4] = fq[1];
    state[5] = fq[2];
    state[6] = fq[3];
}

