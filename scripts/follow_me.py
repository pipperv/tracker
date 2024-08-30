#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import time
import sys
import math
import numpy as np
import smach
import cv2
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped, Twist
import copy
from threading import Thread



import importlib.util
 
# specify the module that needs to be 
# imported relative to the path of the 
# module
spec = importlib.util.spec_from_file_location("tracker","/home/pipe/bender_ws/src/tracker/scripts/tracker.py")

# creates a new module based on spec
tracker = importlib.util.module_from_spec(spec)
 
# executes the module in its own namespace
# when a module is imported or reloaded.
spec.loader.exec_module(tracker)


class Init(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "failed", "timeout"], io_keys=['person_id','tracker'])
    def execute(self,userdata):
        userdata.person_id = 0
        userdata.tracker = tracker.Tracker()
        # userdata.tracker.set_target_id(0)
        # print(f'Following Person with id {0}')
        return 'succeeded'

class NavToTrack(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "failed", "continue", "nav_failed"], io_keys=['person_id','tracker','timer'])
    def execute(self,userdata):
        userdata.tracker.start()
        while ((not (userdata.tracker.get_target_id() is None)) and (not userdata.tracker.is_goal_reached()) and (not userdata.tracker.nav_failed())):
            # id = userdata.tracker.get_target_id()
            # userdata.person_id = id
            # print(id)
            # userdata.tracker.spin()
            userdata.tracker.spin()

        id = userdata.tracker.get_target_id()

        if userdata.tracker.is_goal_reached():
            userdata.tracker.stop()
            return "succeeded"
        elif userdata.tracker.nav_failed():
            userdata.tracker.stop()
            return "nav_failed"
        elif id is None:
            userdata.tracker.stop()
            userdata.timer = time.perf_counter()
            return "failed"
        

class PanHead(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", 'failed', 'timeout'], io_keys=['person_id','tracker','timer'])
        self.pub_vel = rospy.Publisher('/bender/nav/base/cmd_vel', Twist, queue_size=1)
    def execute(self,userdata):
        print('Target not found, looking for target...')
        userdata.tracker.start_search()
        twist = Twist()
        id = userdata.tracker.get_target_id()
        if id is None:
            twist.angular.z = 0.3
            twist.linear.x = 0.0
            self.pub_vel.publish(twist)
            return 'failed'
        elif float(time.perf_counter()) - userdata.timer >= 20:
            userdata.tracker.stop_search()
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            self.pub_vel.publish(twist)
            print('Target Lost...')
            return 'timeout'
        else:
            rospy.sleep(1)
            userdata.tracker.stop_search()
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            self.pub_vel.publish(twist)
            userdata.person_id = id
            userdata.tracker.set_target_id(id)
            rospy.sleep(0.5)
            return 'succeeded'
        
class Recovery(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", 'failed', 'timeout'])
        self.pub_vel = rospy.Publisher('/bender/nav/base/cmd_vel', Twist, queue_size=1)
    def execute(self):
        print('Executing Recovery Behavior...')
        return 'failed'
    
class ReturnInfo(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded"])
    def execute(self):
        print('Ending Follow...')
        return 'succeeded'

def getInstance():

    sm = smach.StateMachine(outcomes=['succeeded', 'failed', 'continue', 'timeout', 'nav_failed'],
                            input_keys=['person_id', 'tracker', 'timer'],
                            output_keys=['person_id', 'tracker', 'timer'])
    
    sm.userdata.person_id = 0

    with sm:

        smach.StateMachine.add('INIT', Init(),
            transitions={
                'succeeded': 'NAV_TO_TRACK'                
            }
        )

        smach.StateMachine.add('NAV_TO_TRACK', NavToTrack(),
            transitions={
                'succeeded': 'RETURN_INFO', 
                'failed': 'PAN_HEAD',
                'continue': 'NAV_TO_TRACK',
                'nav_failed':'RECOVERY'
            }
        )

        smach.StateMachine.add('PAN_HEAD', PanHead(),
            transitions={
                'succeeded': 'NAV_TO_TRACK', 
                'timeout': 'RETURN_INFO',
                'failed': 'PAN_HEAD',
            }
        )

        smach.StateMachine.add('RECOVERY', Recovery(),
            transitions={
                'failed': 'RETURN_INFO'    
            }
        )

        smach.StateMachine.add('RETURN_INFO', ReturnInfo(),
            transitions={
                'succeeded': 'succeeded'    
            }
        )

        return sm

if __name__ == '__main__':

    rospy.init_node('follow_me')

    sm = getInstance()

    outcome = sm.execute() # here is where the test begin
