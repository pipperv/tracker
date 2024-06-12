#!/usr/bin/env python3

import rospy
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from sensor_msgs.msg import Image, PointCloud2, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from dynamixel_workbench_msgs.msg import DynamixelStateList
from dynamixel_workbench_msgs.srv import DynamixelCommand
from cv_bridge import CvBridge

import cv2
import numpy as np
np.float = np.float64
import ros_numpy as rnp
import torch 

from ultralytics import YOLO
from collections import defaultdict
from simple_pid import PID

class Tracker():
    
    def __init__(self):
        torch.cuda.set_device(0) # Set device to gpu
        self.model = YOLO('~/yolov8n-pose.pt') # YOLO model
        self.model.to('cuda')
        self.bridge = CvBridge() # CV bridge
        self.track_history = defaultdict(lambda: []) # Store the track history

        self.pid = PID(1, 0.1, 0.05, setpoint=0)
        self.head_srv = rospy.ServiceProxy('/bender/head/dynamixel_command', DynamixelCommand)
        rospy.sleep(0.3)

        self.current_pan = 512
        self.current_tilt = 490

        self.target_id = 1
        self.dyn_center_pan = 512
        self.dyn_center_tilt = 490

        self.target_frame = "bender/base_link"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.transform = self.tf_buffer.lookup_transform(self.target_frame, "bender/sensors/rgbd_head_depth_optical_frame", rospy.Time(0), rospy.Duration(1.0))

        self.trajectory_msg = JointTrajectory()
        self.trajectory_msg.joint_names = ["head_yaw_joint","head_pitch_joint"]

        self._resolution = None
        self._points_data = None
        self._image_data = None
        self.target_point = PointStamped()

        self.marker_array = MarkerArray()
        self.marker = Marker()
        self.marker.header.frame_id = self.target_frame
        self.marker.header.stamp = rospy.Time.now()
        self.marker.id = 0
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.a = 1.0  # Don't forget to set the alpha!
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0

        self.img_pub = rospy.Publisher("/tracker_result", Image, queue_size=10)
        self.neck_pub = rospy.Publisher("/bender/head/joint_trajectory", JointTrajectory, queue_size=10)
        self.marker_pub = rospy.Publisher("/marker_position", MarkerArray, queue_size=10)

        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1) # Subscriber
        rospy.Subscriber("/bender/sensors/rgbd_head/depth_registered/points", PointCloud2, self.image_callback)
        # rospy.Subscriber("/bender/head/joint_states", JointState, self.joint_callback, queue_size=1)
        rospy.Subscriber("/bender/head/dynamixel_state", DynamixelStateList, self.dyn_state_callback, queue_size=1)

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        try:
            self._points_data = rnp.numpify(msg)
            image_data = self._points_data['rgb'].view(
            (np.uint8, 4))[..., [0, 1, 2]]
            self._image_data = np.ascontiguousarray(image_data)
            self._resolution = self._image_data.shape[:2]

            # # Run YOLOv8 tracking on the frame, persisting tracks between frames
            if not self._image_data is None:
                results = self.model.track(self._image_data, persist=True, classes=[0], tracker="/home/pipe/bender_ws/src/tracker/config/tracker.yaml", verbose=False)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xyn.cpu().tolist()

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            if self.target_id in track_ids:
                for box, track_id, keypoint in zip(boxes, track_ids, keypoints):
                        if track_id == self.target_id:
                            ls_p = keypoint[5] #Right Shoulder Point
                            rs_p = keypoint[6] #Left Shoulder Point
                            x_b, y_b, w_b, h_b = box
                            track = self.track_history[track_id]
                            y_t, x_t = self._resolution[0]*(rs_p[1]+ls_p[1])/2.0, self._resolution[1]*(rs_p[0]+ls_p[0])/2.0 # X Track and Y Track
                            track.append((float(x_t), float(y_t)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)

                            # Draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                            x = self._points_data['x'][int(y_b),int(x_b)]
                            y = self._points_data['y'][int(y_b),int(x_b)]
                            z = self._points_data['z'][int(y_b),int(x_b)]

                            self.target_point.point.x = x
                            self.target_point.point.y = y
                            self.target_point.point.z = z
                            transformed_point = do_transform_point(self.target_point, self.transform)

                            self.publish_marker(transformed_point)

                            if 0.0 in [*ls_p,*rs_p]:
                                self.adjust_neck(x_b, y_b, control_tilt=True)
                            else:
                                self.adjust_neck(x_t, y_t, control_tilt=True)
            elif False:
                pass
            else:
                self.target_id = track_ids[0]
                self.center_neck()

            image_message = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.img_pub.publish(image_message)

        except Exception as e:
            print(e)
            self.center_neck()

    def dyn_state_callback(self, msg):
        self.current_pan = msg.dynamixel_state[1].present_position
        self.current_tilt = msg.dynamixel_state[0].present_position

    def publish_marker(self, point):
        self.marker.pose.position.x = point.point.x
        self.marker.pose.position.y = point.point.y
        self.marker.pose.position.z = point.point.z
        self.marker_array.markers = [self.marker]
        self.marker_pub.publish(self.marker_array)

    def set_neck(self, pan, tilt):
        point = JointTrajectoryPoint()
        point.time_from_start.secs = 0
        point.positions = [pan, tilt]

        rospy.sleep(0.3)
        rospy.loginfo("Publishing")

        self.trajectory_msg.points = [point]

        self.neck_pub.publish(self.trajectory_msg)

    def adjust_neck(self, x, y, control_tilt=False):
        # Pan
        center = self._resolution[1]/2
        error = x - center
        control = int(abs(error/2))
        if error >= 0:
            control = control+1024
        self.head_srv(id=30,   addr_name='Moving_Speed', value=control)
        if control_tilt:
            # Tilt
            center = self._resolution[0]/2
            error = y - center
            control = int(abs(error*2))
            if error <= 0:
                control = control+1024
            self.head_srv(id=31,   addr_name='Moving_Speed', value=control)

    def center_neck(self):
        # Pan Adjust
        try:
            error_pan = self.current_pan - self.dyn_center_pan
            control_pan = int(abs(error_pan*2))
            if error_pan >= 0:
                control_pan = control_pan+1024
            self.head_srv(id=30,   addr_name='Moving_Speed', value=control_pan)
        except Exception as e:
            self.head_srv(id=30,   addr_name='Moving_Speed', value=0)
        rospy.sleep(0.1)
        # Tilt Adjust
        try:
            error_tilt = self.current_tilt - self.dyn_center_tilt
            control_tilt = int(abs(error_tilt))
            if error_tilt >= 0:
                control_tilt = control_tilt+1024
            self.head_srv(id=31,   addr_name='Moving_Speed', value=control_tilt)
        except Exception as e:
            self.head_srv(id=31,   addr_name='Moving_Speed', value=0)
        rospy.sleep(0.1)

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('tracker', anonymous=True)
    try:
        tracker = Tracker()
        
        tracker.spin()
    except rospy.ROSInterruptException:
        pass
