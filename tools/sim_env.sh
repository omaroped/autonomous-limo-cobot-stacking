# Source this in every simulation terminal:   source tools/sim_env.sh
# Puts the simulation on a private ROS channel (this computer only, domain 42), so it cannot
# see or drive a real robot on the default channel (domain 0). The Gazebo launch files refuse
# to start without it. Do not source it in real-robot terminals.
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install/setup.bash"
ros2 daemon stop >/dev/null 2>&1
echo "Simulation terminal: isolated (ROS_LOCALHOST_ONLY=1, ROS_DOMAIN_ID=42)"
