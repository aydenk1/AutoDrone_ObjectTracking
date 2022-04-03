#include "nanodet_openvino.h"
#include "BYTETracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include "std_msgs/String.h"
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "cv-helpers.hpp"
#include "ros/ros.h"
#include <ros/console.h>
#include <iostream>
#include <sstream>

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

struct bbox_tlbr {
    float x1;
    float y1;
    float x2;
    float y2;
};

struct xyz{
    float x;
    float y;
    float z;
};

const float conf_threshold = 0.4;
const float nms_threshold = 0.5;
const bbox_tlbr null_bbox = {.x1=-1.0, .y1=-1.0, .x2=-1.0, .y2=-1.0};
const xyz null_xyz = {.x = -1.0, .y=-1.0, .z=-1.0};
int elapsed_frames = 0; //Modified by update_bbox




void update_bbox_cord(vector<STrack> stracks, bbox_tlbr& old_bbox, bbox_tlbr& new_bbox, xyz& old_cords, xyz& new_cords){
    //Output last detection if within 30 frames, else output null_bbox and null_xyz
    float max_conf = 0;
    new_bbox = null_bbox;
    new_cords = null_xyz;

    //Output last detection if within 30 frames
    if (stracks.size() == 0){
        if(elapsed_frames > 30){
            return;
        }
        ++elapsed_frames;
        new_bbox = old_bbox;
        new_cords = old_cords;
        return;
    }

    //Find max conf bbox from detections
    for (size_t i = 0; i < stracks.size(); i++)
    {
        if (stracks[i].score > max_conf){
            elapsed_frames = 0;
            max_conf = stracks[i].score;

            new_bbox.x1 = stracks[i].tlbr[0];
            new_bbox.y1 = stracks[i].tlbr[1];
            new_bbox.x2 = stracks[i].tlbr[2];
            new_bbox.y2 = stracks[i].tlbr[3];

            old_bbox = new_bbox;
            old_cords = new_cords;
        }
    }
    return;
}




int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;
    int h = src.rows;
    int dst_w = dst_size.width;
    int dst_h = dst_size.height;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;
    float ratio_dst = dst_w * 1.0 / dst_h;

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {
        tmp_w = dst_w;
        tmp_h = floor((dst_w * 1.0 / w) * h);
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

    if (tmp_w != dst_w) {
        int index_w = floor((dst_w - tmp_w) / 2.0);
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {
        int index_h = floor((dst_h - tmp_h) / 2.0);
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
        effect_area.x = 0;
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    return 0;
}


std::vector<float> get_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, object_rect effect_roi)
{
    cv::Mat image = bgr.clone();

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    std::vector<float> boundingboxes;


    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];

        int point = (bbox.x1 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y1 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        point = (bbox.x2 - effect_roi.x) * width_ratio;
        boundingboxes.push_back(point);
        point = (bbox.y2 - effect_roi.y) * height_ratio;
        boundingboxes.push_back(point);
        boundingboxes.push_back(bbox.score * 100);
    }

    return boundingboxes;
}


vector<Object> convert_bytetrack(const std::vector<BoxInfo>& results, const cv::Mat& image, const object_rect effect_roi){
    //Converts Nanodet bbox to ByteTrack style tlwh with score and label
    static int num_dets = results.size();
    vector<Object> objects;
    objects.resize(num_dets);

    int src_w = image.cols;
    int src_h = image.rows;
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;
    for(int i=0; i< results.size(); ++i){
        objects[i].rect.x = (results[i].x1 - effect_roi.x) * width_ratio;
        objects[i].rect.y = (results[i].y1 - effect_roi.y) * height_ratio;
        objects[i].rect.width = ((results[i].x2 - effect_roi.x) * width_ratio) - ((results[i].x1 - effect_roi.x) * width_ratio);
        objects[i].rect.height = ((results[i].y2 - effect_roi.y) * height_ratio) - ((results[i].y1 - effect_roi.y) * height_ratio);
        objects[i].prob = results[i].score;
        objects[i].label = results[i].label;
    }
    return objects;
}


int intelrealsense_inference(ros::Publisher pub_bbox, ros::Publisher pub_rel_pos)
{
    using namespace cv;
    using namespace rs2;
    
    auto detector = NanoDet("/home/px4vision/catkin/src/object_tracking/src/nanodet.xml", "MYRIAD", 32);
    BYTETracker tracker(10, 30);
    const int height = detector.input_size[0];
    const int width = detector.input_size[1];
    const cv::Size model_in_size = cv::Size(width, height);
    bbox_tlbr old_bbox, new_bbox;
    xyz old_cords, new_cords;
    
    ROS_INFO("Model finished loaded\n");


	// Start streaming from Intel RealSense Camera
	ROS_INFO("Setting up camera\n");
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);
	auto const intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    // Init last_bboxes and target_cords to -1
    new_bbox = null_bbox;
    old_bbox = null_bbox;
    new_cords = null_xyz;
    old_cords = null_xyz;

    // Start streaming from Intel RealSense Camera
	ROS_INFO("object_tracking: starting camera stream\n");
    while (ros::ok())
    {
        // Wait for the next set of frames
        rs2::frameset data = pipe.wait_for_frames();
		data = align_to.process(data);

        rs2::video_frame color_frame = data.get_color_frame();
        rs2::depth_frame depth_frame = data.get_depth_frame();

		static int last_frame_number = 0;
		if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = static_cast<int>(color_frame.get_frame_number());
		
        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
		auto depth_mat = depth_frame_to_meters(depth_frame);

        //Perform detection
        cv::Mat resized_img;
        object_rect effect_roi;
        resize_uniform(color_mat, resized_img, cv::Size(width, height), effect_roi);
        auto results = detector.detect(resized_img, conf_threshold, nms_threshold);
        //std::vector<float> bboxes = get_bboxes(color_mat, results, effect_roi);

        //Track objects with bytetrack
        vector<Object> bt_bboxes = convert_bytetrack(results, color_mat, effect_roi);
	    vector<STrack> output_stracks = tracker.update(bt_bboxes);
        

        //Start output logic
        update_bbox_cord(output_stracks, old_bbox, new_bbox, old_cords, new_cords);
		float mean_depth = 0;

        //Get depth info
        if(output_stracks.size() > 0){
			Rect bbox_rect((int)(new_bbox.x1), (int)(new_bbox.y2), (int)(new_bbox.x2 - new_bbox.x1), (int)(new_bbox.y2 - new_bbox.y1));
			bbox_rect = bbox_rect & Rect(0, 0, depth_mat.cols, depth_mat.rows);

            //Find center of bbox
			float pixel[2];
			pixel[0] = new_bbox.x1 + (new_bbox.x2 - new_bbox.x1)/2;
			pixel[1] = new_bbox.y1 + (new_bbox.y2 - new_bbox.y1)/2;
			Scalar m = mean(depth_mat(bbox_rect));
			mean_depth = (float) m[0];

            //Get relative coords
            float point[3];
			rs2_deproject_pixel_to_point(point, &intrinsics, pixel, (float) m[0]);

            //Normalize bbox cords, handle point -> xyz
			new_cords.x = point[0];
			new_cords.y = point[1];
			new_cords.z = point[2];
            old_cords = new_cords;
            new_bbox.x1 /= color_mat.cols;
            new_bbox.y1 /= color_mat.rows;
            new_bbox.x2 /= color_mat.cols;
            new_bbox.y2 /= color_mat.rows;
            old_bbox = new_bbox;
        }

		//ROS_INFO("%f, %f, %f, %f, %d, %d\n", bboxes[0], bboxes[1], bboxes[2], bboxes[3], color_mat.cols, color_mat.rows);
		ROS_INFO("%f, %f, %f, depth: %f", new_cords.x, new_cords.y, new_cords.z, mean_depth);
        geometry_msgs::Quaternion msg;
        msg.x = new_bbox.x1;
        msg.y = new_bbox.y1;
        msg.z = new_bbox.x2;
        msg.w = new_bbox.y2;
        geometry_msgs::QuaternionStamped stamped_msg;
        stamped_msg.header = std_msgs::Header();
        stamped_msg.quaternion = msg;
        pub_bbox.publish(stamped_msg);

		geometry_msgs::Quaternion msg_pos;
        msg_pos.x = new_cords.x;
        msg_pos.y = new_cords.y;
        msg_pos.z = new_cords.z;
        geometry_msgs::QuaternionStamped stamped_msg_pos;
        stamped_msg_pos.header = std_msgs::Header();
        stamped_msg_pos.quaternion = msg_pos;
        pub_rel_pos.publish(stamped_msg_pos);

        ros::spinOnce();
    }
    return 0;
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "object_tracking");
	ROS_INFO("Initiating object-tracking node\n");
    ros::NodeHandle n;
    ros::Publisher pub_bbox = n.advertise<geometry_msgs::QuaternionStamped>("rover/bounding_box",5);
    ros::Publisher pub_rel_pos = n.advertise<geometry_msgs::QuaternionStamped>("rover/rel_pos",5);
    intelrealsense_inference(pub_bbox, pub_rel_pos);
	
	return 0;
}
