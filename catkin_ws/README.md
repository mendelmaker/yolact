# yolact_ros

ROS wrapper for Yolact.

## Installation

Yolact uses Python 3. If you use a ROS version built with Python 2, additional steps are necessary to run the node.

- Set up a Python 3 environment.
- Install the packages required by Yolact. See the Readme on https://github.com/dbolya/yolact for details.
- Additionally, install the packages rospkg and empy in the environment.
- You need to build the cv_bridge module of ROS with Python 3. I recommend using a workspace separate from other ROS packages. Clone the package to the workspace. You might need to adjust some of the following instructions depending on your Python installation.
  ```Shell
  git clone -b melodic https://github.com/ros-perception/vision_opencv.git
  ```
- First method catkin_make,  
  
  - If you use PC catkin_make, compile with
  ```Shell
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  ```
  - If you use Xavier catkin_make, compile with
  ```Shell
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so
  ```
- Second method for script,
  ```Shell
  cd ..
  source how_to_solve_python3_cv2.sh
  ```
- Third method for catkin tools, use
  ```Shell
  catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
  catkin build
  ```
- add the following lines to the postactivate script of your environment (Change the paths according to your workspace path, virtual environment and Python installation):
  ```Shell
  source $HOME/catkin_ws/devel/setup.bash
  export OLD_PYTHONPATH="$PYTHONPATH"
  export PYTHONPATH="$HOME/.virtualenvs/yolact/lib/python3.6/site-packages:$PYTHONPATH"
  ```
- add the following lines to the postdeactivate script of your environment:
  ```Shell
  export PYTHONPATH="$OLD_PYTHONPATH"
