# Figaroh

**updated 07/10**

## scripts/

-   tiago_mocap_calib_def: contains def of functions
-   tiago_mocap_calib_main: configurations generation (for now)
-   tiago_simulate_calibration: LM code for estimation (simulate data + exp data)
-   tiago_simplified: build collision model for tiago (simplified/normal)
-   tools/: dynamic identification tools
-   models/: robots' urdf + meshes
-   mesh_viewer_wrapper: meshcast wrapper class

## basic steps of frame work (draft)

-   Load urdf and create a model for robot (variables: no. of markers on end effector, free-flyer model for base)
-   Build base regressor corresponding to the model selected => base params equations + condition number (select joint offset/full 6-param)
-   Solve IK to generate joint configs (randomness + maximized range + joint limits) -> verify autocollision + condition number
-   Motion planning with Moveit and simulate in Gazebo -> verify the motion
-   Record data with rosbags -> transform to csv by bag2csv code (pay attention to column names)
-   Estimate params with LM code