import xml.etree.ElementTree as ET
import math
import pandas as pd
import os

keyframes = {}
count = 0
# Iter the directory with animation files and get the .xar file for each animation
for root, dirs, files in os.walk("/home/mina/Dropbox/APRIL-MINA/Pre_exp/AnimationsSet_files/plymouth-animations"):
    for fi in files:
        if fi.endswith(".xar"):
            xar = os.path.join(root, fi)
            count += 1
            print ('Count is: ' + str(count))

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
            list_angles_rad_all = []
            list_actuators = []

            # Iterate the actuator curves (a curve per actuator)
            for act_curve in act_list:
                actuator = act_curve.get('actuator')  # Actuator name
                # print(actuator)
                list_actuators.append(actuator)

                list_frames = []
                list_angles_rad = []
                keys = list(act_curve)  # Actuator keyframes
                for key in keys:
                    # Frames must be devided by 1/fps to get timestamp in seconds
                    frame = float(key.get('frame'))
                    frame = round(frame * 0.04, 2)
                    list_frames.append(frame)

                    # Angles are turned into radians
                    angle_deg = float(key.get('value'))
                    angle_rad = round(math.radians(angle_deg), 4)
                    list_angles_rad.append(angle_rad)

                list_frames_all.append(list_frames)
                list_angles_rad_all.append(list_angles_rad)

            # Unique time frames from all actuators. We want one timestamp series for all the actuators
            flat_list = [item for sublist in list_frames_all for item in sublist]
            times_common = sorted(set(flat_list))
            frame_1 = times_common[0] / 2
            times_common.insert(0, frame_1)

            # Dictionary with actuators as keys and list of angles as values
            # Initialize with whole body joint angles for standing posture
            dict_act_ang = {"HeadPitch": [0.0656545], "HeadYaw": [0.0169946], "HipPitch": [-0.0426928],
                            "HipRoll": [-0.00887858],
                            "KneePitch": [-0.00887561], "LElbowRoll": [-0.108104], "LElbowYaw": [-1.71638],
                            "LHand": [0.6942],
                            "LShoulderPitch": [1.77271], "LShoulderRoll": [0.103867], "LWristYaw": [0.0425655],
                            "RElbowRoll": [0.102232], "RElbowYaw": [1.69033], "RHand": [0.688049],
                            "RShoulderPitch": [1.75191],
                            "RShoulderRoll": [-0.0985601], "RWristYaw": [-0.0258008]}

            # Add to the dictionary the rest of joint positionbs for this animation
            for act_a in range(len(list_actuators)):
                dict_act_ang[list_actuators[act_a]].extend(list_angles_rad_all[act_a])

            # Dictionary with actuators as keys and list of frames as values
            dict_act_fra = dict()

            # Add timestamp series for all the actuators that were used for the animation.
            # There might be actuators not added as key-value to this dict
            for act_f in range(len(list_actuators)):
                list_frames_all[act_f].insert(0, frame_1)
                dict_act_fra[list_actuators[act_f]] = list_frames_all[act_f]

            namesJoints = ["HeadYaw", "HeadPitch", "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
                           "LWristYaw",
                           "LHand", "HipRoll", "HipPitch", "KneePitch", "RShoulderPitch", "RShoulderRoll", "RElbowYaw",
                           "RElbowRoll",
                           "RWristYaw", "RHand"]

            # To bring all lists in the same length, the one of the common_times vector
            for item in namesJoints:
                a = dict_act_ang[item]  # Angles dict contains whole body actuators

                # Check if there is a key for the joint in the dict of the frames
                if item in dict_act_fra:
                    f = dict_act_fra[item]
                else:
                    f = [frame_1]

                # Insert time frames and angles to
                for i in range(len(times_common)):
                    if i < len(a):
                        if f[i] != times_common[i]:
                            a.insert(i, a[i - 1])
                            f.insert(i, times_common[i])
                    else:
                        a.append(a[i - 1])
                        f.insert(i, times_common[i])

                dict_act_fra[item] = f
                dict_act_ang[item] = a

            # A dataframe [Joints per Time] with rows as joints (17 namesJoints) and columns are angles per time frame.
            df = pd.DataFrame.from_dict(dict_act_ang)
            df["time"] = times_common
            keyframes[animName] = df

            # dest = '/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/DATA/keyframes/'+animName+'.csv'
            # df.to_csv(dest, sep='\t')


            # Concat all the dataframes in one with an id column containing the name of the animation
            for anim, df in keyframes.iteritems():
                df['id'] = anim
                keyframes[anim] = df
            # Saved at DATA>
            upd = pd.concat(keyframes, ignore_index=True, axis=0)

            # dest = '/home/mina/Dropbox/APRIL-MINA/EXP3_Generation/DATA/df1_KF.csv'
            # upd.to_csv(dest)
