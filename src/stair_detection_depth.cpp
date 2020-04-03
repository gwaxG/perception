#include "perception/stair_detection_depth.h"

struct DepthDetector {

	DepthDetector(int argc, char **argv){
	    pub = n.advertise<std_msgs::String>("chatter", 1000);
	    ros::Rate loop_rate(10);
	    sub = n.subscribe("chatter", 1000, chatterCallback);
        ros::spin();
	}

	void run() {
	    while (1) {
    		cout << "In loop" << endl;
	    }
	}

	static void chatterCallback(const std_msgs::String::ConstPtr& msg)
	{
    	ROS_INFO("I heard u");
	}

    ros::Publisher pub;
    ros::Subscriber sub;
    ros::NodeHandle n;
};



int main(int argc, char **argv)
{
    cout << "indiside" << endl;
    ros::init(argc, argv, "stair_detection_depth");
    DepthDetector(argc, argv);
    return 0;
}
