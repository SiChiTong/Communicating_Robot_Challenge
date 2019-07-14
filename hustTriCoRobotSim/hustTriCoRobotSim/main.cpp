extern "C" {
#include <lua.h>  
#include <lauxlib.h>  
}

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "sol.hpp"
#include "gui.h"
#include "map.h"
#include "control.h"
#include "plan.h"

#include "main.h"

DebugGUI *DebugWindow = NULL;
GlobalMap *RobotMapper = NULL;
RMLCarController *CarController = NULL;
Planner *CarPlanner = NULL;

static double last_sim_time = 0;

bool hust::log(const std::string& str) {
    if (DebugWindow) {
        DebugWindow->log(str);
        return true;
    }
    else {
        return false;
    }
}

bool hust::log_status(const std::string& str) {
    if (DebugWindow) {
        DebugWindow->log_status(str);
        return true;
    }
    else {
        return false;
    }
}

// 显示机器人状态
static void show_robot_status(sol::state_view &lua) {
    Eigen::IOFormat format(Eigen::StreamPrecision, 0, " ", ",", "", "", "[", "]");

    std::ostringstream ss;
    // 输出格式
    ss << std::fixed << std::setprecision(3);

    // 仿真时间
    ss << "Time: ";
    try {
        sol::object sim_time = lua["simGetSimulationTime"]();
        if (sim_time.is<double>()) {
            ss << sim_time.as<double>();
        }
        else {
            throw std::invalid_argument("getSimulationTime() return value is not number");
        }
    }
    catch (const std::exception &err) {
        ss << err.what();
    }

    ss << std::endl;

    // Base 变换矩阵
    ss << "Base: ";
    try {
        sol::object base = lua["hustGetT_WorldBase"]();
        Eigen::Transform<double, 3, Eigen::Affine> trans = hust::convert_to_Affine3d(base);
        Eigen::Vector3d rpy = GetRPYFromRotateMatrix(trans.rotation());
        rpy *= 180 / M_PI;
        ss << "t: " << trans.translation().format(format) << ' ';
        ss << "RPY: " << rpy.format(format);
    }
    catch (const std::exception &err) {
        ss << err.what() << std::endl;
    }
    ss << std::endl;

    // 移动底盘运动速度
    ss << "Base Velocity: ";
    try {
        sol::object base_vel = lua["CarSpeed"];
        auto vel = hust::convert_to_array<double, 3>(base_vel);
        // 世界坐标系下
        ss << "[World] x=" << vel[0] << " y=" << vel[1] << " w=" << vel[2] / M_PI * 180 << " ";
        // 车身坐标系下
        sol::object base = lua["hustGetBase2DPos"]();
        auto base_2d = hust::convert_to_array<double, 3>(base);
        Eigen::Vector2d vel_xy_in_Base = Eigen::Rotation2Dd(-base_2d[2]) * Eigen::Vector2d(vel[0], vel[1]);
        ss << "[Base] x=" << vel_xy_in_Base[0] << " y=" << vel_xy_in_Base[1] << " w=" << vel[2] / M_PI * 180;
    }
    catch (const std::exception &err) {
        ss << err.what() << std::endl;
    }
    ss << std::endl;

    // 末端状态
    //ss << "Left  End: ";
    //try {
    //    sol::object base = lua["hustGetT_WorldBase"]();
    //    Eigen::Transform<double, 3, Eigen::Affine> trans = GetTransFromLua(base);
    //    Eigen::Vector3d rpy = GetRPYFromRotateMatrix(trans.rotation());
    //    rpy *= 180 / M_PI;
    //    ss << "t: " << trans.translation().format(format) << ' ';
    //    ss << "RPY: " << rpy.format(format);
    //}
    //catch (const std::exception &err) {
    //    ss << err.what() << std::endl;
    //}
    //ss << std::endl;

    // 关节状态
    Eigen::Matrix<double, 3, 10> arm_joint_pos;
    auto simGetJointPosition = lua["robot"]["simGetJointPosition"];
    for (int i = 0; i < 10; i++) {
        arm_joint_pos(0, i) = i + 1;
    }
    for (int i = 0; i < 8; i++) {
        double pos = simGetJointPosition(lua["arm"]["left"][i+1]);
        arm_joint_pos(1, i) = pos / M_PI * 180;
    }
    for (int i = 8; i < 10; i++) {
        double pos = simGetJointPosition(lua["arm"]["left"][i+1]);
        arm_joint_pos(1, i) = pos;
    }
    for (int i = 0; i < 8; i++) {
        double pos = simGetJointPosition(lua["arm"]["right"][i+1]);
        arm_joint_pos(2, i) = pos / M_PI * 180;
    }
    for (int i = 8; i < 10; i++) {
        double pos = simGetJointPosition(lua["arm"]["right"][i+1]);
        arm_joint_pos(2, i) = pos;
    }
    ss << "Arm Joint Position (1-8: /deg, 9-10: /m): " << std::endl;
    ss << arm_joint_pos.format(Eigen::IOFormat()) << std::endl;

    hust::log_status(ss.str());
}

// 距离传感器
struct ProxSensor {
    std::string name;           // 传感器名称
    Eigen::Affine3d rel_pos;    // 传感器位姿变换
    double min_radius;          // 最小测距
    double max_radius;          // 最大测距
    double range_rad;           // 感应角度范围
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

static const std::vector<ProxSensor, Eigen::aligned_allocator<ProxSensor>> ProxSensorParam = {
{ "laserF", Eigen::Translation3d(0, 0, 0.1) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0, 2, 0 },
{ "laserL", Eigen::Translation3d(0, 0, 0.1) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0, 2, 0 },
{ "laserB", Eigen::Translation3d(0, 0, 0.1) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0, 2, 0 },
{ "laserR", Eigen::Translation3d(0, 0, 0.1) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0, 2, 0 },
{ "proxL", Eigen::Translation3d(0, 0, 0.08) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0.1, 0.7, M_PI / 2 },
{ "proxR", Eigen::Translation3d(0, 0, 0.08) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0.1, 0.7, M_PI / 2 },
{ "proxB", Eigen::Translation3d(0, 0, 0.08) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0.1, 0.7, M_PI / 2 },
{ "proxF", Eigen::Translation3d(0, 0, 0.08) * Eigen::AngleAxis<double>(-M_PI / 2, Eigen::Vector3d(0, 1, 0)), 0.1, 0.9, M_PI / 2 },
};

// 感知更新
extern "C" static int sensor_update(lua_State *L) {
    sol::state_view lua(L);

    sol::object lua_task_status = lua["TaskStatus"];
    std::vector<int> task_status = hust::convert_to_vector<int>(lua_task_status);

    // 传感器更新地图

    // 获取传感器位姿
    for (const ProxSensor& sensor : ProxSensorParam) {
        int sensor_id = lua[sensor.name];    // 传感器ID
        sol::object lua_trans = lua["robot"]["GetObjectMatrix"](sensor_id);
        Eigen::Affine3d T_WorldSensorHeadZ = hust::convert_to_Affine3d(lua_trans);
        Eigen::Affine3d T_WorldSensorHeadX = T_WorldSensorHeadZ * sensor.rel_pos;
        // 获取检测结果
        sol::object occupied, detecte_point, detected_object_handle;
        sol::tie(occupied, detecte_point, detected_object_handle) = lua["robot"]["simGetProximitySensorResult"](sensor_id);
        // 计算末端坐标
        Eigen::Vector2d end;
        if (occupied.as<int>() == 1) {
            std::array<double, 3> pos3d = hust::convert_to_array<double, 3>(detecte_point);
            Eigen::Vector3d T_WorldEnd = T_WorldSensorHeadZ * Eigen::Vector3d(pos3d[0], pos3d[1], pos3d[2]);
            end = Eigen::Vector2d(T_WorldEnd[0], T_WorldEnd[1]);
        }
        // 投影到二维平面
        Eigen::Affine2d start = Eigen::Affine2d(Eigen::Translation2d(T_WorldSensorHeadX.translation()[0], T_WorldSensorHeadX.translation()[1]));
        start = start.rotate(GetRPYFromRotateMatrix(T_WorldSensorHeadX.rotation())[2]);

        // 获取障碍物信息
        int object_handle = detected_object_handle.as<int>();
        GlobalMap::ObjectType object_type = GlobalMap::ObjectType::UNKNOWN;
        Eigen::Affine3d object_pos3d;
        Eigen::Affine2d object_pos2d;
        if (occupied.as<int>() == 1) {
            // 获取障碍物位姿
            sol::object object_pos_trans = lua["robot"]["GetObjectMatrix"](detected_object_handle);
            object_pos3d = hust::convert_to_Affine3d(object_pos_trans);
            object_pos2d = Eigen::Affine2d(Eigen::Translation2d(object_pos3d.translation()[0], object_pos3d.translation()[1]));
            object_pos2d = object_pos2d.rotate(GetRPYFromRotateMatrix(object_pos3d.rotation())[2]);
            // 检查障碍物类型
            bool is_task2_object = false;
            sol::object is_task2_object_lua = lua["robot"]["ifHandleIsTask2Obj"](detected_object_handle);
            if (is_task2_object_lua.is<bool>() && is_task2_object_lua.as<bool>()) {
                is_task2_object = true;
            }
            bool is_task3_object = false;
            sol::object is_task3_object_lua = lua["robot"]["ifHandleIsGasTank"](detected_object_handle);
            if (is_task3_object_lua.is<bool>() && is_task3_object_lua.as<bool>()) {
                is_task3_object = true;
            }
            if (is_task2_object) {
                object_type = GlobalMap::ObjectType::FURNITURE;
            }
            else if (is_task3_object) {
                object_type = GlobalMap::ObjectType::GASTANK;
            }
            else {
                object_type = GlobalMap::ObjectType::UNKNOWN;
            }
        }

        bool blocked = false;
        if (task_status[0] == 2 && task_status.size() > 1) {
            std::string carry_name = lua["Task2Furnitures"][task_status[1]]["name"];
            if (object_type == GlobalMap::ObjectType::UNKNOWN) {
                // 正在搬运椅子, 禁止前部接近传感器检测未知障碍物
                blocked = true;
            }
        }
        
        if (!blocked) {
            // 更新全局地图
            RobotMapper->update(
                occupied.as<int>() == 1,
                start, end,
                sensor.min_radius, sensor.max_radius, sensor.range_rad,
                object_type, object_handle, object_pos3d
            );
        }
    }

    // 更新已发现的任务二家具位姿
    for (const int& fur_id : RobotMapper->get_furniture_ids()) {
        sol::object lua_trans = lua["robot"]["GetObjectMatrix"](fur_id);
        Eigen::Affine3d pos3d = hust::convert_to_Affine3d(lua_trans);
        // 更新家具信息
        RobotMapper->update_furniture_pos(fur_id, pos3d);
    }

    // 导出检测到任务二家具信息到lua
    lua["Task2Furnitures"] = lua.create_table();
	try {
		auto furs = RobotMapper->get_furnitures_area();
		for (int quadrant = 1; quadrant < 5; quadrant++) {
			if (furs[quadrant - 1] == NULL) {
				lua["Task2Furnitures"][quadrant] = sol::nil;
				hust::log_status("[FUR " + std::to_string(quadrant) + "] unknwon");
			}
			else {
				// 类型
				FurnitureMap::FurnitureType type = furs[quadrant - 1]->check_type();
				std::string type_name = FurnitureMap::get_name_by_type(type);
				if (type != FurnitureMap::FurnitureType::UNKNOWN) {
					lua["Task2Furnitures"][quadrant] = lua.create_table();
					// ID
					lua["Task2Furnitures"][quadrant]["id"] = furs[quadrant - 1]->id;
					// 类型
					lua["Task2Furnitures"][quadrant]["name"] = type_name;
					// 位姿
					Eigen::Affine2d pos = furs[quadrant - 1]->get_universal_pos2d();
					lua["Task2Furnitures"][quadrant]["pos"] = lua.create_table();
					lua["Task2Furnitures"][quadrant]["pos"][1] = pos.translation()[0];
					lua["Task2Furnitures"][quadrant]["pos"][2] = pos.translation()[1];
					lua["Task2Furnitures"][quadrant]["pos"][3] = hust::get_angle(pos);
					// 是否在目标区域
					bool in_dst = furs[quadrant - 1]->is_in_dst_area();
					lua["Task2Furnitures"][quadrant]["in_dst"] = in_dst;
					// 调试输出
					std::ostringstream ss;
					ss << "[FUR " << quadrant << "]";
					ss << "ID=" << furs[quadrant - 1]->id << ", ";
					ss << "type=" << type_name << ", ";
					ss << "pos=(" << pos.translation()[0] << ", " << pos.translation()[1] << ", " << hust::get_angle(pos) << "), ";
					ss << "in_dst=" << in_dst;
					hust::log_status(ss.str());
				}
				else {
					// 调试输出
					std::ostringstream ss;
					ss << "[FUR " << quadrant << "]";
					ss << "ID=" << furs[quadrant - 1]->id << ", ";
					ss << "type=" << type_name;
					hust::log_status(ss.str());
				}
			}
		}
	}
	catch (const std::exception& err) {
		// 调试输出
		std::ostringstream ss;
		ss << "[FUR] Failed to detect furnitures position: " << err.what();
		hust::log_status(ss.str());
	}

     // 导出检测到的任务三障碍物信息到lua
    lua["Task3Obstacles"] = lua.create_table();
    auto obs = RobotMapper->get_ordered_task3_obs();
    std::ostringstream ss;
    ss << "[OBS]";
    for (int i = 1; i <= obs.size(); i++) {
        lua["Task3Obstacles"][i] = lua.create_table();
        // ID
        lua["Task3Obstacles"][i]["id"] = obs[i - 1].id;
        // 位姿
        Eigen::Affine2d pos = OccupyMap::pos3d_to_2d(obs[i - 1].get_pos());
        lua["Task3Obstacles"][i]["pos"] = lua.create_table();
        lua["Task3Obstacles"][i]["pos"][1] = pos.translation()[0];
        lua["Task3Obstacles"][i]["pos"][2] = pos.translation()[1];
        lua["Task3Obstacles"][i]["pos"][3] = hust::get_angle(pos);
        // 期望绕行方向
        lua["Task3Obstacles"][i]["clockwise"] = (i % 2 == 0);
        // 调试输出
        ss << " " << i << "={ID=" << obs[i].id << ", ";
        ss << "pos=(" << pos.translation()[0] << ", " << pos.translation()[1] << ", " << hust::get_angle(pos) << "), ";
        ss << "clockwise=" << (i % 2 == 1) << "}";
    }
    hust::log_status(ss.str());

    // 更新已发现的任务四煤气罐位姿
    try {
        int id = RobotMapper->get_gas_tank_id();
        sol::object lua_trans = lua["robot"]["GetObjectMatrix"](id);
        Eigen::Affine3d pos3d = hust::convert_to_Affine3d(lua_trans);
        RobotMapper->update_gas_tank_pos(pos3d);

        // 导出检测到任务四家具信息到lua
        lua["Task4GasTank"] = lua.create_table();
        // ID
        lua["Task4GasTank"]["id"] = id;
        // 位姿
        Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(pos3d);
        lua["Task4GasTank"]["pos"] = lua.create_table();
        lua["Task4GasTank"]["pos"][1] = pos2d.translation()[0];
        lua["Task4GasTank"]["pos"][2] = pos2d.translation()[1];
        lua["Task4GasTank"]["pos"][3] = hust::get_angle(pos2d);
        // 是否在任务四目标区域
        bool in_task4_dst = RobotMapper->is_gas_task_in_task4_dst_area();
        lua["Task4GasTank"]["in_task4_dst"] = in_task4_dst;
        // 是否在任务五目标区域
        bool in_task5_dst = RobotMapper->is_gas_task_in_task5_dst_area();
        lua["Task4GasTank"]["in_task5_dst"] = in_task5_dst;
        // 调试输出
        std::ostringstream ss;
        ss << "[GAS TANK]";
        ss << "ID=" << id << ", ";
        ss << "pos=(" << pos2d.translation()[0] << ", " << pos2d.translation()[1] << ", " << hust::get_angle(pos2d) << "), ";
        ss << "in_task4_dst=" << in_task4_dst << ", ";
        ss << "in_task5_dst=" << in_task5_dst;
        hust::log_status(ss.str());
    }
    catch (const std::out_of_range&) {
        lua["Task4GasTank"] = sol::nil;
        hust::log_status("[GAS TANK] unknwon");
    }

    // 更新地图中的车辆位置
    do {
        // 获取车辆位置
        sol::object lua_trans = lua["hustGetT_WorldBase"]();
        Eigen::Affine3d T_WorldBase = hust::convert_to_Affine3d(lua_trans);
        // 投影到二维平面
        Eigen::Affine2d pos = Eigen::Affine2d(Eigen::Translation2d(T_WorldBase.translation()[0], T_WorldBase.translation()[1]));
        pos = pos.rotate(GetRPYFromRotateMatrix(T_WorldBase.rotation())[2]);
        DebugWindow->set_car_position(pos);
    } while (0);

	return 0;	// 设置返回值数量
}

// 移动平台规划更新
static int task3_direction; // 1 逆时针绕行开始 0 禁用绕行 -1 顺时针绕行开始
extern "C" static int car_plan_update(lua_State *L) {
    sol::state_view lua(L);

    sol::object lua_task_status = lua["TaskStatus"];
    std::vector<int> task_status = hust::convert_to_vector<int>(lua_task_status);

    if (task_status[0] == 2 && task_status.size() > 1) {
        // 在导航地图中排除正在搬运的物体
        int carry_id = lua["Task2Furnitures"][task_status[1]]["id"];
        RobotMapper->set_plan_map_ignores({ carry_id });
    }
    else if (task_status[0] == 4 || task_status[0] == 5) {
        try {
            int gas_tank_id = RobotMapper->get_gas_tank_id();
            RobotMapper->set_plan_map_ignores({ gas_tank_id });
        }
        catch (const std::exception& ) {
            RobotMapper->set_plan_map_ignores({});
        }
    }
    else {
        RobotMapper->set_plan_map_ignores({});
    }

    std::string plan_mode = lua["CarPlanMode"];

    if (plan_mode.substr(0, 4) == "plan") {

        // 获取车辆位置
        sol::object lua_trans = lua["hustGetT_WorldBase"]();
        Eigen::Affine3d T_WorldBase = hust::convert_to_Affine3d(lua_trans);
        Eigen::Affine2d car_pos = OccupyMap::pos3d_to_2d(T_WorldBase);

        // 获取车辆速度
        sol::object lua_car_speed = lua["CarSpeed"];
        std::array<double, 3> car_speed = hust::convert_to_array<double, 3>(lua_car_speed);

        // 获取目标位置
        sol::object lua_target_pos = lua["CarPlanGoal"];
        if (!lua_target_pos.is<sol::table>()) {
            // 禁用了导航
            lua["CarPlanResultTrajectory"] = sol::nil;
            return 0;
        }
        std::array<double, 3> target_pos_array = hust::convert_to_array<double, 3>(lua_target_pos);
        Eigen::Affine2d target_pos = Eigen::Translation2d(target_pos_array[0], target_pos_array[1]) * Eigen::Rotation2Dd(target_pos_array[2]);

        // 读取最大加速度与最大速度
        double max_accel_linear = lua["FastMaxBrake"];
        double max_vel_linear = lua["FastMaxVel"];
        double max_accel_rad = lua["FastMaxBrakeRot"];
        double max_vel_rad = lua["FastMaxVelRot"];

        int wait_count = 0;
        bool has_switch_dir = false;
        auto plan_start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            if (task_status[0] == 1) {
                // 任务一插入额外的障碍
                std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> addition_lines;
                // 任务二起始区域右侧
                addition_lines.emplace_back(FurnitureMap::task2_src_right_top);
                addition_lines.emplace_back(FurnitureMap::task2_src_right_top[0], FurnitureMap::task2_src_left_bottom[1]);
                // 任务二起始区域下部
                addition_lines.emplace_back(FurnitureMap::task2_src_right_top[0], FurnitureMap::task2_src_left_bottom[1]);
                addition_lines.emplace_back(FurnitureMap::task2_src_left_bottom);

                RobotMapper->set_plan_addition_lines(addition_lines);
            }
            else if (task_status[0] == 3) {

                if (std::chrono::high_resolution_clock::now() - plan_start_time > std::chrono::seconds(5) && !has_switch_dir) {
                    // 超过5秒仍然无法得到路径, 切换绕行方向
                    task3_direction = -task3_direction;
                    has_switch_dir = true;
                    hust::log("[PLAN] Detour, another side");
                }
                else if (std::chrono::high_resolution_clock::now() - plan_start_time > std::chrono::seconds(10)) {
                    // 超过10秒仍然无法得到路径, 禁用绕行
                    task3_direction = 0;
                    hust::log("[PLAN] Disable detour");
                }

                if (task3_direction != 0) {
                    int first_clockwise;
                    if (task3_direction > 0) {
                        first_clockwise = 0;
                    }
                    else {
                        first_clockwise = 1;
                    }
                    // 任务三插入额外障碍
                    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> addition_lines;
                    // 读取任务三障碍物, 需要加锁
                    std::lock_guard<std::mutex> lock(RobotMapper->mutex_obstacles);
                    auto obs = Obstacle::get_ordered_task3_obstacles(RobotMapper->obstacles);

					// 首个障碍物默认为逆时针绕行, 在其下方插入虚拟墙
					if (obs.size() >= 1) {
						Eigen::Vector2d pos_this = OccupyMap::pos3d_to_2d(obs[0].get_pos()).translation();
						addition_lines.emplace_back(pos_this);
						addition_lines.emplace_back(Eigen::Vector2d(pos_this[0], FurnitureMap::task2_dst_left_bottom[1]));
					}

					// 第二个障碍默认顺时针绕行, 在其上方插入虚拟墙
					if (obs.size() >= 2) {
						Eigen::Vector2d pos_this = OccupyMap::pos3d_to_2d(obs[1].get_pos()).translation();
						addition_lines.emplace_back(pos_this);
						addition_lines.emplace_back(Eigen::Vector2d(pos_this[0], FurnitureMap::task2_src_right_top[1]));
					}

					// 第三个障碍默认逆时针绕行, 在其下方插入虚拟墙
					if (obs.size() >= 3) {
						// 检查第二个与第三个障碍之间的间距
						Eigen::Vector2d pos_second = OccupyMap::pos3d_to_2d(obs[1].get_pos()).translation();
						Eigen::Vector2d pos_this = OccupyMap::pos3d_to_2d(obs[2].get_pos()).translation();
						if ((pos_second - pos_this).norm() < Obstacle::task3_obs_min_distance) {
							// 距离过近不绕行, 添加引导墙
							// todo
						}
						else {
							addition_lines.emplace_back(pos_this);
							addition_lines.emplace_back(Eigen::Vector2d(pos_this[0], FurnitureMap::task2_dst_left_bottom[1]));
						}
					}

					// 第四个障碍默认顺时针绕行, 在其上方插入虚拟墙
					if (obs.size() >= 4) {
						Eigen::Vector2d pos_this = OccupyMap::pos3d_to_2d(obs[3].get_pos()).translation();
						// 检查第四个障碍物到墙之间的距离
						if (std::abs(FurnitureMap::task2_src_right_top[1] - pos_this[1]) < Obstacle::task3_obs_wall_min_distance) {
							// 距离过近, 不绕行, 添加引导墙
							// todo
						}
						else {
							addition_lines.emplace_back(pos_this);
							addition_lines.emplace_back(Eigen::Vector2d(pos_this[0], FurnitureMap::task2_src_right_top[1]));
						}
					}

					// 首个障碍物默认为逆时针绕行, 在其下方插入虚拟墙
					if (obs.size() >= 5) {
						Eigen::Vector2d pos_this = OccupyMap::pos3d_to_2d(obs[4].get_pos()).translation();
						addition_lines.emplace_back(pos_this);
						addition_lines.emplace_back(Eigen::Vector2d(pos_this[0], FurnitureMap::task2_dst_left_bottom[1]));
					}

					RobotMapper->set_plan_addition_lines(addition_lines);
                }
                else {
                    // 禁用绕行
                    RobotMapper->set_plan_addition_lines({});
                }
            }
            else {
                RobotMapper->set_plan_addition_lines({});
            }

            std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> trajectory;
            // 获取路径
            trajectory = CarPlanner->get_trajectory(
                car_pos, target_pos, car_speed,
                max_accel_linear, max_accel_rad,
                max_vel_linear, max_vel_rad
            );


            if (!trajectory.empty()) {

                sol::table lua_traj = lua.create_table();
                // 转换为lua中的轨迹
                for (const HeadTrajectoryPoint& p : trajectory) {
                    sol::table entry = lua.create_table();
                    entry["pos"] = lua.create_table();
                    entry["pos"][1] = p.pos.translation()[0];
                    entry["pos"][2] = p.pos.translation()[1];
                    entry["pos"][3] = Eigen::Rotation2Dd(p.pos.rotation()).angle();
                    entry["vel"] = lua.create_table();
                    entry["vel"][1] = p.vel_linear[0];
                    entry["vel"][2] = p.vel_linear[1];
                    entry["vel"][3] = p.vel_rad;
                    entry["length"] = p.length;
                    entry["time"] = p.time;
                    entry["curvature"] = p.curvature;
                    lua_traj.add(entry);
                }
                lua["CarPlanResultTrajectory"] = lua_traj;
                break;
            }
            wait_count++;
            if (wait_count == 1) {
                hust::log("[PLAN] Waiting for trajectory...");
            }
        }
        if (wait_count > 0) {
            hust::log("[PLAN] Found trajectory");
        }
    }

    return 0;
}

// 移动平台控制更新
extern "C" static int car_control_update(lua_State *L) {
    sol::state_view lua(L);

    // 加速度, 速度限制
    CarController->set_max_accel(
        hust::convert_to_array<double, 3>(sol::object(lua["RMLCarMaxAccel"])));
    CarController->set_max_vel(
        hust::convert_to_array<double, 3>(sol::object(lua["RMLCarMaxVel"])));
    // 期望位姿速度
    CarController->set_target_pos_vel(
        hust::convert_to_array<double, 3>(sol::object(lua["RMLCarTrgtPos"])),
        hust::convert_to_array<double, 3>(sol::object(lua["RMLCarTrgtVel"])));
    // 当前位姿, 计算控制
    int ret = CarController->update(
        hust::convert_to_array<double, 3>(sol::object(lua["RMLCarCurPos"])));
    // 控制的输出
    std::array<double, 3> out_vel = CarController->get_output_vel();
    lua["RMLCarOutVel"] = lua.create_table();
    lua["RMLCarOutVel"][1] = out_vel[0];
    lua["RMLCarOutVel"][2] = out_vel[1];
    lua["RMLCarOutVel"][3] = out_vel[2];

    // 记录控制状态
    hust::log_status(RMLCarController::get_return_string(ret));
    if (ret == ReflexxesAPI::RML_WORKING) {
    }
    else if (ret == ReflexxesAPI::RML_FINAL_STATE_REACHED) {
    }
    else {
        hust::log(RMLCarController::get_return_string(ret));
    }
    return 0;
}

// 手动控制与界面显示数据
extern "C" static int debug_gui_update(lua_State *L) {
    sol::state_view lua(L);

    show_robot_status(lua);
    // 执行控制指令
    auto arm_cmd = DebugWindow->get_cmds();
    for (auto cmd : arm_cmd) {
        lua.script(cmd);
    }
    return 0;
}

// 完成该控制周期
extern "C" static int step_finish(lua_State *L) {
    sol::state_view lua(L);
    // 在调试界面上打印调试信息
    DebugWindow->clear_log_status();

    // 检查仿真频率是否稳定
    double now_time = lua["simGetSimulationTime"]();
    if (last_sim_time != 0) {
        double time_step = now_time - last_sim_time;
        if (time_step > 0.05 * 1.01) {
            hust::log("Bad TimeStep:" + std::to_string(time_step));
            lua["simExtPrintInfo"]("Bad TimeStep:" + std::to_string(time_step));
        }
    }
    last_sim_time = now_time;

    // 执行仿真
    lua["simSwitchThread"]();
    return 0;
}

extern "C" static int lua_log(lua_State *L) {
    const char *str = NULL;
    size_t str_size = 0; //Lua strings have an explicit length; they can contain null characters.

    if (lua_gettop(L) != 1)
    {
        lua_pushstring(L, "Must provide string parameter to this hust.log()");
        lua_error(L);
    }
    str = luaL_checklstring(L, 1, &str_size);

    if (str && str_size > 0) {
        hust::log(std::string(str, str_size));
        sol::state_view lua(L);
        lua["simExtPrintInfo"](std::string(str, str_size));
    }
    return 0;
}

extern "C" static int lua_log_status(lua_State *L) {
    const char *str = NULL;
    size_t str_size = 0; //Lua strings have an explicit length; they can contain null characters.

    if (lua_gettop(L) != 1)
    {
        lua_pushstring(L, "Must provide string parameter to this hust.log_status()");
        lua_error(L);
    }
    str = luaL_checklstring(L, 1, &str_size);

    if (str && str_size > 0) {
        hust::log_status(std::string(str, str_size));
    }
    return 0;
}

// 要导出到Lua中的函数
static const struct luaL_Reg hustTriCoRobotSim[] = {
    { "sensor_update", sensor_update },
    { "car_plan_update", car_plan_update },
    { "car_control_update", car_control_update },
    { "debug_gui_update", debug_gui_update },
    { "step_finish", step_finish },
    { "log", lua_log },
    { "log_status", lua_log_status },
	{ NULL, NULL }
};

extern "C" __declspec(dllexport) int luaopen_hustTriCoRobotSim(lua_State *L) {
    sol::state_view lua(L);

	// 注册要导入到Lua中的函数
	luaL_register(L, "hustTriCoRobotSim", hustTriCoRobotSim);

    // 二维变换 * 二维变换
    lua.set_function("hustXYWMulXYW", [](sol::this_state L, sol::object lua_a, sol::object lua_b) -> sol::table {
        sol::state_view lua(L);
        auto arr_a = hust::convert_to_array<double, 3>(lua_a);
        auto arr_b = hust::convert_to_array<double, 3>(lua_b);
        Eigen::Affine2d a = hust::xyw_to_affine2d(arr_a);
        Eigen::Affine2d b = hust::xyw_to_affine2d(arr_b);
        Eigen::Affine2d res = a * b;
        auto arr_res = hust::affine2D_to_xyw(res);
        sol::table lua_res = lua.create_table();
        lua_res[1] = arr_res[0];
        lua_res[2] = arr_res[1];
        lua_res[3] = arr_res[2];
        return lua_res;
    });

    // 二维变换 * 二维向量
    lua.set_function("hustXYWMulVec", [](sol::this_state L, sol::object lua_a, sol::object lua_b) -> sol::table {
        sol::state_view lua(L);
        auto arr_a = hust::convert_to_array<double, 3>(lua_a);
        auto arr_b = hust::convert_to_array<double, 2>(lua_b);
        Eigen::Affine2d a = hust::xyw_to_affine2d(arr_a);
        Eigen::Vector2d b(arr_b[0], arr_b[1]);
        Eigen::Vector2d res = a * b;
        sol::table lua_res = lua.create_table();
        lua_res[1] = res[0];
        lua_res[2] = res[1];
        return lua_res;
    });

    // 环境初始化
    std::string cwd = lua["hustCWD"];

    // 地图构建器
    if (RobotMapper == NULL) {
        RobotMapper = new GlobalMap(
            Eigen::Vector2d(-14, -15), 
            Eigen::Vector2d(4, 4), 0.02, 
            cwd + "HUST/map_",
            cwd + "HUST/map_origin.png");
        CarController = new RMLCarController();
    }
    else {
        RobotMapper->reset();
    }

    // GUI界面
    if (DebugWindow == NULL) {
        DebugWindow = new DebugGUI();

        DebugWindow->set_map(RobotMapper);
        DebugWindow->set_map_save_callback(std::bind(&GlobalMap::save_all, RobotMapper, cwd + "map", "png"));

        DebugWindow->set_car_size(Eigen::Vector2d(0.8, 0.6));
        DebugWindow->set_car_position(Eigen::Transform<double, 2, Eigen::Affine>(Eigen::Translation2d(-3.3, 1.722)));

        DebugWindow->run_in_thread();
    }
    DebugWindow->clear_log();
    DebugWindow->clear_log_status();

    // 车辆路径规划
    hust::log("Initializing CarPlanner...");
    if (CarPlanner != NULL) {
        DebugWindow->set_planner(NULL);
        CarPlanner->stop_planning();
        delete CarPlanner;
        CarPlanner = NULL;
    }
    CarPlanner = new Planner(*RobotMapper, cwd + "msvc-ws/motion primitive/task2.mprim");
    hust::log("Initializing CarPlanner Finish");
    // 默认逆时针绕行
    task3_direction = 1;

    DebugWindow->set_planner(CarPlanner);

    //CarPlanner->set_goal(Eigen::Affine2d(Eigen::Translation2d(-3.3, 1.722)));
    //CarPlanner->set_goal(Eigen::Translation2d(0.218, -4.31) * Eigen::Rotation2Dd(-M_PI / 2));
    //CarPlanner->set_goal(Eigen::Translation2d(-1.098, -5.638) * Eigen::Rotation2Dd(M_PI / 2));
    CarPlanner->start_plannig();

    // 车辆运动控制
    if (CarController != NULL) {
        delete CarController;
    }
    hust::log("Initializing CarController...");
    CarController = new RMLCarController(0.05);
    hust::log("Initializing CarController Finish");

    // 避免程序波动, 禁止自动切换线程
    lua["simSetThreadSwitchTiming"](200);
    lua["simSetThreadAutomaticSwitch"](false);
    last_sim_time = 0;

    // 打印加载成功
    lua["simExtPrintInfo"](std::string("hustTriCoRobotSim.dll (compiled at ") + __DATE__ + " " + __TIME__ + ") load OK");
    hust::log("LUA Library OK");
	return 1;
}

