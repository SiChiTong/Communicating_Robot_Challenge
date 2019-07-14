#pragma once

#include <vector>
#include <string>

#include <ReflexxesAPI.h>
#include <RMLPositionFlags.h>
#include <RMLPositionInputParameters.h>
#include <RMLPositionOutputParameters.h>

class RMLCarController {
private:
    double timestep;
    bool first_step = true;
    ReflexxesAPI rml;
    RMLPositionInputParameters  ip;
    RMLPositionOutputParameters op;
    RMLPositionFlags flags;
public:
    static std::string get_return_string(int code) {
        switch (code) {
        case ReflexxesAPI::RML_WORKING:
            return std::to_string(code) + " RML_WORKING";
        case ReflexxesAPI::RML_FINAL_STATE_REACHED:
            return std::to_string(code) + " RML_FINAL_STATE_REACHED";
        case ReflexxesAPI::RML_ERROR:
            return std::to_string(code) + " RML_ERROR";
        case ReflexxesAPI::RML_ERROR_INVALID_INPUT_VALUES:
            return std::to_string(code) + " RML_ERROR_INVALID_INPUT_VALUES";
        case ReflexxesAPI::RML_ERROR_EXECUTION_TIME_CALCULATION:
            return std::to_string(code) + " RML_ERROR_EXECUTION_TIME_CALCULATION";
        case ReflexxesAPI::RML_ERROR_SYNCHRONIZATION:
            return std::to_string(code) + " RML_ERROR_SYNCHRONIZATION";
        case ReflexxesAPI::RML_ERROR_NUMBER_OF_DOFS:
            return std::to_string(code) + " RML_ERROR_NUMBER_OF_DOFS";
        case ReflexxesAPI::RML_ERROR_NO_PHASE_SYNCHRONIZATION:
            return std::to_string(code) + " RML_ERROR_NO_PHASE_SYNCHRONIZATION";
        case ReflexxesAPI::RML_ERROR_NULL_POINTER:
            return std::to_string(code) + " RML_ERROR_NULL_POINTER";
        case ReflexxesAPI::RML_ERROR_EXECUTION_TIME_TOO_BIG:
            return std::to_string(code) + " RML_ERROR_EXECUTION_TIME_TOO_BIG";
        case ReflexxesAPI::RML_ERROR_USER_TIME_OUT_OF_RANGE:
            return std::to_string(code) + " RML_ERROR_USER_TIME_OUT_OF_RANGE";
        default:
            return std::to_string(code) + " unknown";
        }
    }

    RMLCarController(double timestep=0.05) :
        timestep(timestep), rml(3, timestep), ip(3), op(3)
    {
        // 选择自由度
        ip.SelectionVector->VecData[0] = true;
        ip.SelectionVector->VecData[1] = true;
        ip.SelectionVector->VecData[2] = true;
        // 仅时间同步
        flags.SynchronizationBehavior = RMLPositionFlags::PHASE_SYNCHRONIZATION_IF_POSSIBLE;
        //flags.SynchronizationBehavior = RMLPositionFlags::NO_SYNCHRONIZATION;
    }

    void set_max_vel(const std::array<double, 3> &max_vel = { 1.0f / 0, 1.0f / 0, 1.0f / 0 }) {
        ip.MaxVelocityVector->VecData[0] = max_vel.at(0);
        ip.MaxVelocityVector->VecData[1] = max_vel.at(1);
        ip.MaxVelocityVector->VecData[2] = max_vel.at(2);
    }

    void set_max_accel(const std::array<double, 3> &max_accel = { 1.0f / 0, 1.0f / 0, 1.0f / 0 }) {
        ip.MaxAccelerationVector->VecData[0] = max_accel.at(0);
        ip.MaxAccelerationVector->VecData[1] = max_accel.at(1);
        ip.MaxAccelerationVector->VecData[2] = max_accel.at(2);
    }

    void set_target_pos_vel(const std::array<double, 3> &target_pos, const std::array<double, 3> &target_vel = {0, 0, 0}) {
        // 目标位置
        ip.TargetPositionVector->VecData[0] = target_pos.at(0);
        ip.TargetPositionVector->VecData[1] = target_pos.at(1);
        ip.TargetPositionVector->VecData[2] = target_pos.at(2);
        // 目标速度
        ip.TargetVelocityVector->VecData[0] = target_vel.at(0);
        ip.TargetVelocityVector->VecData[1] = target_vel.at(1);
        ip.TargetVelocityVector->VecData[2] = target_vel.at(2);
    }

    int update(const std::array<double, 3> &cur_pos) {
        std::array<double, 3> cur_vel;
        if (first_step) {
            cur_vel = { 0, 0, 0 };
            first_step = false;
        }
        else {
            // 若未给定速度, 则进行差分计算
            cur_vel = {
                (cur_pos.at(0) - ip.CurrentPositionVector->VecData[0]) / timestep,
                (cur_pos.at(1) - ip.CurrentPositionVector->VecData[1]) / timestep,
                0,
            };
            // 需要考虑角度环形关节
            double delta_rad = cur_pos.at(2) - ip.CurrentPositionVector->VecData[2];
            while (delta_rad > M_PI) {
                delta_rad -= 2 * M_PI;
            }
            while (delta_rad < -M_PI) {
                delta_rad += 2 * M_PI;
            }
            cur_vel[2] = delta_rad / timestep;
        }

        return update(cur_pos, cur_vel);
    }
    int update(const std::array<double, 3> &cur_pos, const std::array<double, 3> &cur_vel) {
        // 当前位置
        ip.CurrentPositionVector->VecData[0] = cur_pos.at(0);
        ip.CurrentPositionVector->VecData[1] = cur_pos.at(1);
        ip.CurrentPositionVector->VecData[2] = cur_pos.at(2);
        // 当前速度
        ip.CurrentVelocityVector->VecData[0] = cur_vel.at(0);
        ip.CurrentVelocityVector->VecData[1] = cur_vel.at(1);
        ip.CurrentVelocityVector->VecData[2] = cur_vel.at(2);
        // 角度为环形关节, 优先走角度差较小的方向
        double delta_rad = ip.TargetPositionVector->VecData[2] - ip.CurrentPositionVector->VecData[2];
        while (delta_rad > M_PI) {
            delta_rad -= 2 * M_PI;
        }
        while (delta_rad < -M_PI) {
            delta_rad += 2 * M_PI;
        }
        ip.TargetPositionVector->VecData[2] = delta_rad + ip.CurrentPositionVector->VecData[2];

        // 调用在线轨迹生成算法, 进行位置闭环控制
        return rml.RMLPosition(ip, &op, flags);
    }

    // 获取控制输出: 当前期望速度
    std::array<double, 3> get_output_vel() {
        std::array<double, 3> out_vel = {
            op.NewVelocityVector->VecData[0],
            op.NewVelocityVector->VecData[1],
            op.NewVelocityVector->VecData[2], };
        return out_vel;
    }

    // 获取控制输出: 当前期望位置
    std::array<double, 3> get_output_pos() {
        std::array<double, 3> out_pos = {
            op.NewPositionVector->VecData[0],
            op.NewPositionVector->VecData[1],
            op.NewPositionVector->VecData[2], };
        return out_pos;
    }
};
