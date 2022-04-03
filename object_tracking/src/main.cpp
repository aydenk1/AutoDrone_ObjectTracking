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

typedef struct bbox_tlbr {
    float x1 = -1;
    float y1 = -1;
    float x2 = -1;
    float y2 = -1;
} bbox_tlbr;

typedef struct xyz{
    float x = -1;
    float y = -1;
    float z = -1;
} xyz;

vector<bbox_tlbr> get_bboxes(vector<STrack> stracks){
    //Convert bytetrack stracks to vector of tlbr bboxex
    vector<bbox_tlbr> bboxes_tlbr;
    bboxes_tlbr.resize(stracks.size());
    for (size_t i = 0; i < stracks.size(); i++)
    {
        bboxes_tlbr[i].x1 = stracks[i].tlbr[0];
        bboxes_tlbr[i].y1 = stracks[i].tlbr[1];
        bboxes_tlbr[i].x2 = stracks[i].tlbr[2];
        bboxes_tlbr[i].y2 = stracks[i].tlbr[3];
    }
    return bboxes_tlbr;
}

inline void copy_points(float* src_points, float* dst_points){
    dst_points[0] = src_points[0];
    dst_points[1] = src_points[1];
    dst_points[2] = src_points[2];
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
    const float conf_threshold = 0.4;
    const float nms_threshold = 0.5;
    ROS_INFO("Model finished loaded\n");


	// Start streaming from Intel RealSense Camera
	ROS_INFO("Setting up camera\n");
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);
	auto const intrinsics = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();

    // Init last_bboxes and target_cords to zero
    int count = 0;
    vector<bbox_tlbr> old_bboxes[1];
	vector<xyz> old_cords[1];

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
        std::vector<Object> bt_bboxes = convert_bytetrack(results, color_mat, effect_roi);
		vector<STrack> output_stracks = tracker.update(bt_bboxes);
        vector<bbox_tlbr> bboxes = get_bboxes(output_stracks);

        //Init xyz coords
        vector<xyz> cords;
		float mean_depth = 0;
        if(bboxes.size() == 0) {
            bboxes = old_bboxes;
			cords = old_cords;
			count++;
			if(count > 30) {
				bboxes[0] = -1;
				bboxes[1] = -1;
				bboxes[2] = -1;
				bboxes[3] = -1;
				
				point[0] = -1;
				point[1] = -1;
				point[2] = -1;
			}
        }
        else {
			Rect object((int)(bboxes[0]), (int)(bboxes[3]),
						(int)(bboxes[2]-bboxes[0]),
						(int)(bboxes[3]-bboxes[1]));
			object = object & Rect(0, 0, depth_mat.cols, depth_mat.rows);
			float pixel[2];
			pixel[0] = bboxes[0] + (bboxes[2]-bboxes[0])/2;
			pixel[1] = bboxes[1] + (bboxes[3]-bboxes[1])/2;
			Scalar m = mean(depth_mat(object));
			mean_depth = (float) m[0];
			rs2_deproject_pixel_to_point(point, &intrinsics, pixel, (float) m[0]);
			old_point[0] = point[0];
			old_point[1] = point[1];
			old_point[2] = point[2];

            bboxes[0] /= color_mat.cols;
            bboxes[1] /= color_mat.rows;
            bboxes[2] /= color_mat.cols;
            bboxes[3] /= color_mat.rows;
            old_bboxes = bboxes;
			count = 0;
        }

		//ROS_INFO("%f, %f, %f, %f, %d, %d\n", bboxes[0], bboxes[1], bboxes[2], bboxes[3], color_mat.cols, color_mat.rows);
		ROS_INFO("%f, %f, %f, depth: %f", point[0], point[1], point[2], mean_depth);
        geometry_msgs::Quaternion msg;
        msg.x = bboxes[0];
        msg.y = bboxes[1];
        msg.z = bboxes[2];
        msg.w = bboxes[3];
        geometry_msgs::QuaternionStamped stamped_msg;
        stamped_msg.header = std_msgs::Header();
        stamped_msg.quaternion = msg;
        pub_bbox.publish(stamped_msg);

		geometry_msgs::Quaternion msg_pos;
        msg_pos.x = point[0];
        msg_pos.y = point[1];
        msg_pos.z = point[2];
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
