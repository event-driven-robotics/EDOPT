#pragma once

#include <yarp/os/LogComponent.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/Network.h>
#include <yarp/os/Node.h>
#include <yarp/os/Publisher.h>
#include <yarp/os/Time.h>
// #include <yarp/rosmsg/std_msgs/Int64.h>
// #include <yarp/rosmsg/geometry_msgs/Pose.h>
#include <yarp/rosmsg/SharedData.h>

using yarp::os::Network;
using yarp::os::Node;
using yarp::os::Publisher;

namespace {
    YARP_LOG_COMPONENT(TALKER, "yarp.example.ros.talker")
    constexpr double loop_delay = 0.1;
}

class YarpToRos
{

private:

public:

    // yarp::os::Publisher<yarp::rosmsg::geometry_msgs::Pose> publisher;
    yarp::os::Publisher<yarp::rosmsg::SharedData> port;  // changed Port to Publisher

    yarp::os::Node* node = nullptr;

    // yarp::rosmsg::geometry_msgs::Pose data;

    yarp::rosmsg::SharedData d;

    void initPublisher(){
       // Network yarp;
        /* creates a node called /yarp/talker */
        node = new yarp::os::Node("/yarp/talker");

       if (!port.topic("foo2/sharedmessage"))              // replaced open() with topic()
       {
           yCError(TALKER) << "Failed to create publisher to /position";
       }

        /* subscribe to topic chatter */
        // if (!publisher.topic("/star_position")) {
        //     yCError(TALKER) << "Failed to create publisher to /catcher/event/object_position";
        // }
    }

    void publishTargetPos(double x, double y, double z, double qx, double qy, double qz, double qw){

        d.content.push_back(x);
        d.content.push_back(y);
        d.content.push_back(z);
        d.content.push_back(qx);
        d.content.push_back(qy);
        d.content.push_back(qz);
        d.content.push_back(qw);
        yInfo()<<d.content;
        // std::cout<<std::endl; 

        port.write(d);

        d.content.clear();

        /* publish it to the topic */
        // publisher.write(data);
    }

    ~YarpToRos()
    {
        if (node)
        {
            delete node;
            node = nullptr;
        }
    }

};

