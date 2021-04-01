# Installation
Unzip this stack on the ROS WORKSPACE and build it
~~~~bash
pip install gym tqdm sharedmem
catkin build cbn_irl gym_2d
~~~~



# Training / Testing commands for BN-IRL in a circular wall environment
rosrun cbn_irl bnirl_2d_circle --tr --renew --viz --irl_renew --eta 0.1
rosrun cbn_irl bnirl_2d_circle --te --viz --eta 0.1

# Training / Testing commands for CBN-IRL in a circular wall environment
rosrun cbn_irl cbnirl_2d_circle --tr --renew --viz --irl_renew --eta 0.1
rosrun cbn_irl cbnirl_2d_circle --te --viz --eta 0.1




