# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import os

from dataloader.sample import Sample

# ----------------------------------------------------------------------------------------------------------------------
gnames = [
    'cartwheel_left',
    'cartwheel_right',
    'duck',
    'hook_left',
    'hook_right',
    'jab_left',
    'jab_right',
    'kick_left',
    'kick_right',
    'push',
    'sidekick_left',
    'sidekick_right',
    'uppercut_left',
    'uppercut_right'
]
gnames_beautified = [
    'Cartwheel (left)',
    'Cartwheel (right)',
    'Duck',
    'Hook (left)',
    'Hook (right)',
    'Jab (left)',
    'Jab (right)',
    'Kick (left)',
    'Kick (right)',
    'Push',
    'Side kick (left)',
    'Side kick (right)',
    'Uppercut (left)',
    'Uppercut (right)'
]

#
# Joint indices
#
(HIP_CENTER,
 SPINE_CENTER,
 NECK,
 HEAD,

 SHOULDER_LEFT,
 ELBOW_LEFT,
 WRIST_LEFT,
 HAND_LEFT,

 SHOULDER_RIGHT,
 ELBOW_RIGHT,
 WRIST_RIGHT,
 HAND_RIGHT,

 HIP_LEFT,
 KNEE_LEFT,
 ANKLE_LEFT,
 FOOT_LEFT,
 HIP_RIGHT,
 KNEE_RIGHT,
 ANKLE_RIGHT,
 FOOT_RIGHT,

 SPINE_TOP,

 JOINT_CNT) = range(22)  # dataset has 21 joints

#
# Skeleton structure
#
BONES = [
    # upper body
    (HIP_CENTER, SPINE_CENTER),
    (SPINE_CENTER, SPINE_TOP),
    (SPINE_TOP, NECK),
    (NECK, SHOULDER_LEFT),
    (NECK, SHOULDER_RIGHT),
    (SHOULDER_LEFT, ELBOW_LEFT),
    (SHOULDER_RIGHT, ELBOW_RIGHT),
    (ELBOW_LEFT, WRIST_LEFT),
    (ELBOW_RIGHT, WRIST_RIGHT),
    (WRIST_LEFT, HAND_LEFT),
    (WRIST_RIGHT, HAND_RIGHT),

    (NECK, HEAD),

    # lower body
    (HIP_CENTER, HIP_LEFT),
    (HIP_CENTER, HIP_RIGHT),
    (HIP_LEFT, KNEE_LEFT),
    (HIP_RIGHT, KNEE_RIGHT),
    (KNEE_LEFT, ANKLE_LEFT),
    (KNEE_RIGHT, ANKLE_RIGHT),
    (ANKLE_LEFT, FOOT_LEFT),
    (ANKLE_RIGHT, FOOT_RIGHT),
]


# ----------------------------------------------------------------------------------------------------------------------
def load_jk2017(root):
    """
    Loads the JK2017 (Kinect) dataset.

    :param root: the root path of the data
    :return: a list of Sample objects
    """
    root = os.path.join(root, 'kinect', 'training')

    samples = []

    for sname in sorted(os.listdir(root)):
        # Build subject's path
        spath = os.path.join(root, sname)

        for gname in sorted(os.listdir(spath)):
            # Build the gesture's path for this subject
            gpath = os.path.join(spath, gname)

            for ename in sorted(os.listdir(gpath)):
                # Build the example's path for this subject
                epath = os.path.join(gpath, ename)

                # Parse the example file
                with open(epath) as f:
                    split = f.read().split('####')
                    # Make sure the file's label matches with the directory's name
                    parsed_gname, parsed_len = split[0].strip().splitlines()
                    assert parsed_gname == gname

                    frames = split[1:]
                    pts = []

                    # Parse each frame's data
                    for frame in frames:
                        joints_str = (','.join(frame.strip().splitlines())).split(',')
                        pts += [np.asarray([float(joint) for joint in joints_str], dtype=np.float)]

                    # Make sure we parsed the correct number of frames
                    assert len(pts) == int(parsed_len)

                    # Make a sample out of this
                    sample = Sample(pts, gnames_beautified[gnames.index(gname)])
                    samples += [sample]

    return samples

# ----------------------------------------------------------------------------------------------------------------------
