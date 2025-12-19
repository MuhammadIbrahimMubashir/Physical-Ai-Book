# Chapter 4: Robot Description (URDF)

## What is URDF?
URDF stands for **Unified Robot Description Format**.  
It is a file that describes:
- Robot shape  
- Joints  
- Links (body parts)  
- Sensors  

URDF tells ROS 2 and simulation software **how the robot is built**.

---

## Links and Joints
1. **Link**
   - A link is a single part of the robot  
   - Example: Arm, leg, or torso  

2. **Joint**
   - A joint connects two links  
   - Allows movement (rotation or sliding)  

Example: Shoulder joint connects upper arm to torso.

---

## Visual vs Collision Models
- **Visual model:** What the robot looks like  
- **Collision model:** Used for physics simulation to avoid objects passing through each other  

---

## Why URDF is Important
- Lets ROS 2 understand your robot  
- Helps simulate robot in Gazebo  
- Important for controlling humanoid robots  

---

## Summary
- URDF describes robot shape and structure  
- Links = body parts, joints = connections  
- Visual and collision models help simulation
