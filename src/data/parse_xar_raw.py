import xml.etree.ElementTree as ET
import math
import pandas as pd
import os
from settings import *

"""
Creates a dataframe with the raw data extracted from the animation .xar files
"""
anim_dir = os.path.join(ROOT_PATH, ANIM_DIR)
dest = os.path.join(ROOT_PATH, RAW_DATA, 'df10_KF.csv')

keyframes = {}
count = 0
# Iter the directory with animation files and get the .xar file for each animation
for root, dirs, files in os.walk(anim_dir):
    for fi in files:
        if fi.endswith(".xar"):
            xar = os.path.join(root, fi)
            count += 1
            print('Count is: ' + str(count))

            # Open the XML .xar file and save content as string
            with open(xar, 'r') as myfile:
                data = myfile.read().replace('\n', '')

            # xmlns namespace causes problems in search the Element tree. Needs to be deleted
            ns_old = '<ChoregrapheProject xmlns="http://www.aldebaran-robotics.com/schema/choregraphe/project.xsd" xar_version="3">'
            ns_new = '<ChoregrapheProject xar_version="3">'

            # If the problematic substring is there delete it
            if ns_old in data:
                data = data.replace(ns_old, ns_new)
            # Parse the xmlstring into a tree
            tree = ET.ElementTree(ET.fromstring(data))

            # Get the fps attribute value
            fps = tree.find(".//*[@fps]")
            fps = float(fps.get('fps'))
            print('Fps is: ' + str(fps))

            # Get the name of the animation
            animName = tree.find(".//Diagram/Box")
            animName = animName.get('name')
            print('Animation is: ' + animName)

            # Get a list of the actuators
            act_list = tree.find(".//ActuatorList")
            list_frames_all = []
            list_angles_deg_all = []
            list_actuators = []

            # Iterate the actuator curves (a curve per actuator)
            for act_curve in act_list:
                actuator = act_curve.get('actuator')  # Actuator name
                # print(actuator)
                list_actuators.append(actuator)

                list_frames = []
                list_angles_deg = []
                keys = list(act_curve)  # Actuator keyframes
                for key in keys:
                    # Frames must be multiplied by 1/fps to get timestamp in seconds
                    frame = int(key.get('frame'))
                    list_frames.append(frame-1)

                    # Angles
                    angle_deg = float(key.get('value'))
                    list_angles_deg.append(angle_deg)

                list_frames_all.append(list_frames)
                list_angles_deg_all.append(list_angles_deg)

            # DEBUGGING
            if len(list_actuators) < 17:
                print(list_actuators)
                print('list_actuators is of size ' + str(len(list_actuators)))

            # Dictionary with actuators as keys and list of angles as values
            dict_act_ang = dict((k, []) for k in joints_names)

            # Add to the dictionary the joint positions (in degrees) for this animation
            for act_a in range(len(list_actuators)):
                dict_act_ang[list_actuators[act_a]].extend(list_angles_deg_all[act_a])


            try:
                # One animation df just timestep x joints
                df = pd.DataFrame.from_dict(dict_act_ang)

                # The keyframes numbers for this animation
                times = [list(i) for i in set(map(tuple, list_frames_all))]
                if len(times) == 1:
                    # Add the number of appearence for keyframes in this animation df
                    df["keyframe"] = list_frames_all[0]
                    df['id'] = animName

                    # Add df as value yto the keyframes dictionary
                    keyframes[animName] = df

                else:
                    print("Some problem with timeframes. Are there joint values missing?")
                    print("Animation : " + animName)
            except:
                print("Animation: " + xar)

# # Concat all the dataframes in one with an id column containing the name of the animation
# for anim, df in keyframes.items():
#     df['id'] = anim
#     keyframes[anim] = df

upd = pd.concat(keyframes, ignore_index=True, axis=0)
upd.to_csv(dest)
