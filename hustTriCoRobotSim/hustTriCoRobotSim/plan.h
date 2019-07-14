#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <mutex>
#include <condition_variable>
#include <iostream>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <sbpl/headers.h>

#include "main.h"
#include "map.h"

// [有头模式] 轨迹点, 带有位置和速度信息
class HeadTrajectoryPoint {
public:
    Eigen::Affine2d pos;        // 轨迹点位姿
    double curvature;           // 曲率 (曲率半径的倒数)
    double length;              // 到达该点的总路径长度
    Eigen::Vector2d vel_linear; // 期望线速度
    double vel_rad;             // 期望角速度
    double time;                // 预计到达该点的时间
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    HeadTrajectoryPoint(
        const Eigen::Affine2d& pos = Eigen::Affine2d(),
        double curvature = std::numeric_limits<double>::signaling_NaN(),
        double length = -1,
        const Eigen::Vector2d& vel_linear = Eigen::Vector2d(0, 0),
        double vel_rad = 0,
        double time = -1
    ) {
        this->pos = pos;
        this->curvature = curvature;
        this->length = length;
        this->vel_linear = vel_linear;
        this->vel_rad = vel_rad;
        this->time = time;
    }
};

class Planner {
public:
    GlobalMap &global_map;
    const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> car;
    std::vector<sbpl_2Dpt_t> car_perimeter;
    double car_min_radius = 0;
    double car_max_radius = 0;
    const int angle_number = 16;

    Eigen::Affine2d accurate_start;       // 精确的起点
    Eigen::Affine2d accurate_goal;        // 精确的终点
    std::array<int, 3> grid_start = { 0, 0, 0 };        // 网格的起点
    std::array<int, 3> grid_goal = { 0, 0, 0 };         // 网格的终点

    cv::Mat map;                // 导航地图
    EnvironmentNAVXYTHETALAT *env = NULL;
    SBPLPlanner *planner = NULL;

    std::thread plan_thread;

    // 上一次的可行轨迹
    // 当目标点被改变, 会被重置
    // 当地图改变后, 会被重置
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> last_available_trajectory;

    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> temp_trajectory;
    Eigen::Vector2d temp_cls_point;

    std::mutex mutex_buffer;
    /* 以下变量需要加锁访问 */
    bool stop_plan = false;
    Eigen::Affine2d buffer_accurate_start;       // 精确的起点
    Eigen::Affine2d buffer_accurate_goal;        // 精确的终点
    std::array<int, 3> buffer_grid_start = { 0, 0, 0 };
    std::array<int, 3> buffer_grid_goal = {0, 0, 0};
    std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> solution_backend;
    std::condition_variable cv_has_solution;

    int last_plan_map_id = -1;       // 上一次从GlobalMap获取到的用于规划的导航地图id
    int last_check_map_id = -1;      // 上一次检测没问题的地图id

    double safe_percent = 1;
    const double min_safe_percent_generate = 1.0;
    const double min_safe_percent_test = 0.90;

public:
    
    // 计算点到最近的路径点
    static bool nearest_path_point(
        const Eigen::Affine2d pos,
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        int* p_nearest_next = NULL,     // 下一个路径点
        double* p_err_linear = NULL,    // 位置差
        double* p_err_rad = NULL        // 角度差
    ) noexcept
    {
        if (path.empty()) {
            // 路径为空
            return false;
        }

        int nearest_next;
        double err_linear = std::numeric_limits<double>::infinity();
        double err_rad;

        Eigen::Vector2d car_xy = pos.translation();
        double car_angle = Eigen::Rotation2Dd(pos.rotation()).angle();

        if (path.size() == 1) {
            // 轨迹长度为 1
            nearest_next = 0;
            err_linear = (car_xy - path[0].translation()).norm();
            err_rad = car_angle - Eigen::Rotation2Dd(path[0].rotation()).angle();
        }
        else {
            // 轨迹长度 >= 2
            // 获取与当前车辆位置最近的轨迹线段, 并记录位置偏差
            for (int i = path.size() - 1; i > 0; i--) {
                // 前后路径点
                Eigen::Affine2d next = path[i];
                Eigen::Affine2d prev = path[i - 1];
                // 路径点坐标
                Eigen::Vector2d next_xy = next.translation();
                Eigen::Vector2d prev_xy = prev.translation();
                // 路径线段到车身的直线距离
                double line_percent;
                double distance = min_dist_to_line_seg(car_xy, prev_xy, next_xy, NULL, &line_percent);
                if (distance < err_linear) {
                    // 记录位置偏差
                    nearest_next = i;
                    err_linear = distance;
                    // 计算路径线段上的角度差
                    double next_angle = Eigen::Rotation2Dd(path[i].rotation()).angle();
                    double prev_angle = Eigen::Rotation2Dd(path[i - 1].rotation()).angle();
                    double diff_angle = next_angle - prev_angle;
                    while (diff_angle > M_PI)
                        diff_angle -= 2 * M_PI;
                    while (diff_angle < -M_PI)
                        diff_angle += 2 * M_PI;
                    // 计算车辆角度与轨迹的差
                    err_rad = car_angle - (prev_angle + diff_angle * line_percent);
                }
            }
        }
        // 角度差, 限制为[-pi, pi)
        while (err_rad > M_PI)
            err_rad -= 2 * M_PI;
        while (err_rad < -M_PI)
            err_rad += 2 * M_PI;

        if (p_nearest_next) {
            *p_nearest_next = nearest_next;
        }
        if (p_err_linear) {
            *p_err_linear = err_linear;
        }
        if (p_err_rad) {
            *p_err_rad = err_rad;
        }
        return true;
    }

    // 简化路径, 合并纯自转与同方向的平移
    static std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> simplify_path(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        bool only_move = false,     // 无视旋转
        double tol_linear = 0.001, 
        double tol_rad = 0.1 / 180 * M_PI)
    {
        if (path.size() <= 2) {
            // 路径长度过短, 无需简化
            return path;
        }

        std::vector<int> new_index;

        // 简化后的路径的起点与原始路径相同
        new_index.push_back(0);

        bool merge_rotate = false;   // 正在合并旋转
        bool merge_move = false;     // 正在合并平移
        for (size_t i = 1; i < path.size(); i++) {
            const Eigen::Affine2d& prev = path[new_index.back()];
            const Eigen::Affine2d& cur = path[i];

            double err_linear = (prev.translation() - cur.translation()).norm();
            double err_rad = std::abs(hust::diff_angle(prev, cur));

            if (!merge_rotate && !merge_move) {
                // 还未发生合并
                if (err_linear < tol_linear && err_rad < tol_rad)
                {
                    // 无旋转无平移
                    // 跳过该点
                }
                else if (err_rad >= tol_rad && err_linear < tol_linear) {
                    // 仅旋转
                    if (!only_move) {
                        merge_rotate = true;
                    }
                } 
                else if (err_linear >= tol_linear && err_rad < tol_rad) {
                    // 仅平移
                    merge_move = true;
                }
                else {
                    // 旋转+平移
                    if (!only_move) {
                        new_index.push_back(i);
                    }
                    else {
                        merge_move = true;
                    }
                }
            }
            else if (merge_rotate) {
                // 正在合并旋转
                if (err_linear >= tol_linear) {
                    // 有平移, 回退
                    i--;    
                    new_index.push_back(i);
                    merge_rotate = false;
                }
                else {
                    // 无平移
                }
            }
            else if (merge_move) {
                // 正在合并平移
                if (err_rad >= tol_rad && !only_move) {
                    // 有旋转, 回退
                    i--;
                    new_index.push_back(i);
                    merge_move = false;
                }
                else {
                    // 判断点是否都在连线上
                    bool all_in_line = true;
                    for (size_t j = new_index.back() + 1; j < i; j++) {
                        double dist = min_dist_to_line_seg(
                            path[j].translation(),
                            path[new_index.back()].translation(),
                            path[i].translation()
                        );
                        if (dist > tol_linear) {
                            all_in_line = false;
                            break;
                        }
                    }
                    if (all_in_line) {
                        // 所有点在连线上
                    }
                    else {
                        // 有点在连线外, 回退
                        i--;
                        new_index.push_back(i);
                        merge_move = false;
                    }
                }
            }
        }
        // 简化后的路径的终点与原始路径相同
        if (new_index.back() != path.size() - 1)
        {
            new_index.push_back(path.size() - 1);
        }

        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> new_path;
        new_path.reserve(new_index.size());
        for (const int& i : new_index) {
            new_path.push_back(path[i]);
        }

        return new_path;
    }

    // 验证某个坐标点是否自由
    bool is_pixel_free(const Eigen::Vector2d& p, uint8_t occupied_value = OccupyMap::PIXEL_NOT_PLAN) {
        // 转换为地图坐标
        Eigen::Vector2i p_map = global_map.convert_to_map_pos(p);
        // 验证点是否在地图内
        if (!global_map.check_position(p_map) ||
            map.at<uint8_t>(cv::Point(p_map[0], p_map[1])) >= 0xFF - occupied_value
        ) {
            // 内插点无效或被占据 圆弧无效
            return false;
        }
        return true;
    }

    // 验证某一条直线上的所有坐标点是否自由
    bool is_line_free(
        const Eigen::Vector2d& front, const Eigen::Vector2d& back,
        uint8_t occupied_value = OccupyMap::PIXEL_NOT_PLAN, Eigen::Vector2d* bad_pos = NULL, 
        double interplote_interval = -1
    ) {
        if (!is_pixel_free(front, occupied_value)) {
            if (bad_pos != NULL) {
                *bad_pos = front;
            }
            return false;
        } else if (!is_pixel_free(back, occupied_value)) {
            if (bad_pos != NULL) {
                *bad_pos = back;
            }
            return false;
        }
        if (interplote_interval <= 0) {
            interplote_interval = global_map.resolution;
        }
        Eigen::Vector2d line = back - front;
        double distance = line.norm();
        int interplote_count = static_cast<int>(std::ceil(distance / interplote_interval)) - 1;
        for (int i = 0; i < interplote_count; i++) {
            double percent = (i + 1.0f) / (interplote_count + 1);
            Eigen::Vector2d p = front + percent * line;
            if (!is_pixel_free(p, occupied_value)) {
                if (bad_pos != NULL) {
                    *bad_pos = p;
                }
                return false;
            }
        }
        return true;
    }

    // 验证某一条直线上的所有位姿是否可行
    bool is_line_passable(
        const Eigen::Affine2d& front, const Eigen::Affine2d& back, 
        bool ignore_angle = false, Eigen::Vector2d* cls_point = NULL, 
        double interplote_interval = -1
    ) {
        if (!is_pos_passable(front, cls_point) || !is_pos_passable(back, cls_point)) {
            return false;
        }
        if (interplote_interval <= 0) {
            interplote_interval = global_map.resolution;
        }
        Eigen::Vector2d line = back.translation() - front.translation();
        double line_angle = std::atan2(line[1], line[0]);
        double distance = line.norm();
        double diff_angle = hust::diff_angle(front, back);
        int interplote_count = static_cast<int>(std::ceil(distance / interplote_interval)) - 1;
        for (int i = 0; i < interplote_count; i++) {
            double percent = (i + 1.0f) / (interplote_count + 1);
            Eigen::Vector2d p = front.translation() + percent * line;
            double angle;
            if (ignore_angle) {
                angle = line_angle;
            }
            else {
                angle = hust::get_angle(front) + percent * diff_angle;
            }
            Eigen::Affine2d pos = Eigen::Translation2d(p) * Eigen::Rotation2Dd(angle);
            if (!is_pos_passable(pos, cls_point)) {
                return false;
            }
        }
        return true;
    }

    // 验证某个位姿是否可行 , 验证机器人的周长是否被占据, 每隔分辨率取一点
    bool is_pos_passable(const Eigen::Affine2d& car_pos, Eigen::Vector2d* bad_pos = NULL) {
        //if (!is_pixel_free(car_pos.translation(), OccupyMap::PIXEL_NOT_PLAN)) {
        //    if (bad_pos != NULL) {
        //        *bad_pos = car_pos.translation();
        //    }
        //    return false;
        //}
        for (int i = 0; i < (int)car.size(); i++) {
            Eigen::Vector2d p0 = car_pos * (car[i] * safe_percent);
            Eigen::Vector2d p1 = car_pos * (car[(i + 1) % car.size()] * safe_percent);
            if (!is_line_free(p0, p1, OccupyMap::PIXEL_LIMITED, bad_pos)) {
                return false;
            }
        }
        return true;
    }

    // 输入两个圆上带有方向的点, 计算是否为圆弧是否可行
    // 两点方向变化量应 < 180°
    bool is_curve_passable(
        // 输入参数
        const Eigen::Affine2d& arc_front, const Eigen::Affine2d& arc_back, 
        // 输出参数
        double& arc_radius, Eigen::Vector2d& arc_center, double& arc_interplote_interval, 
        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& arc_interplote, 
        Eigen::Vector2d* bad_pos = NULL
    ) {
        // 验证圆弧起点与终点是否有效
        if (!is_pos_passable(arc_front, bad_pos) || !is_pos_passable(arc_back, bad_pos)) {
            return false;
        }

        // 角度变化量 +-
        double diff_angle = hust::sub_2pi(hust::get_angle(arc_front), hust::get_angle(arc_back));
        // 弦长
        double chord_length = (arc_back.translation() - arc_front.translation()).norm();
        // 曲率半径 +-, 左转半径为正, 右转为负 
        double radius = (chord_length / 2) / std::sin(diff_angle / 2);
        // 曲率圆心
        Eigen::Vector2d center = arc_front * Eigen::Vector2d(0, radius);
        // 圆弧开始角度
        Eigen::Vector2d c_to_p0 = arc_front.translation() - center;
        double begin_angle = std::atan2(c_to_p0[1], c_to_p0[0]);
        // 圆弧结束角度
        Eigen::Vector2d c_to_p1 = arc_back.translation() - center;
        double end_angle = std::atan2(c_to_p1[1], c_to_p1[0]);
        // 弧长
        double arc_length = std::abs(diff_angle * radius);
        // 圆弧内插点数
        int interplote_count = static_cast<int>(arc_length / global_map.resolution);
        // 初始化内插点数组
        arc_interplote.clear();
        // 在插值点上, 验证是否在地图上有碰撞
        bool is_collision = false;
        for (int interplote_i = 1; interplote_i <= interplote_count; interplote_i++) {
            double percent = 1.0 * interplote_i / (interplote_count + 1);
            double angle_of_circle = begin_angle + percent * (end_angle - begin_angle);     // 相对于圆心的角度
            double angle_of_forward = hust::get_angle(arc_front) + percent * diff_angle;    // 车头朝向
            // 要验证的点的全局坐标
            Eigen::Vector2d p = center + Eigen::Vector2d(std::cos(angle_of_circle), std::sin(angle_of_circle)) * std::abs(radius);
            Eigen::Affine2d pos = Eigen::Translation2d(p) * Eigen::Rotation2Dd(angle_of_forward);
            if (is_pos_passable(pos, bad_pos)) {
                // 内插点有效
                arc_interplote.push_back(pos);
            }
            else {
                // 内插点在地图外或被占据 圆弧无效
                is_collision = true;
                break;
            }
        }
        if (!is_collision) {
            // 无碰撞, 为有效的圆弧
            arc_radius = radius;
            arc_center = center;
            arc_interplote_interval = arc_length / (interplote_count + 1);
            return true;
        }
        else {
            return false;
        }
    }

    // 检查某一个拐点能否使用圆弧拟合
    bool turn_has_passable_curve(
        // 输入参数
        const Eigen::Vector2d& turn_point,
        double line_front_angle, double line_back_angle, 
        double max_shorten, double min_shorten, 
        // 输出参数
        Eigen::Affine2d& arc_front, Eigen::Affine2d& arc_back, 
        double& arc_radius, Eigen::Vector2d& arc_center, 
        double& arc_interplote_interval, 
        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& arc_interplote, 
        Eigen::Vector2d* cls_point = NULL
    ) {
        int shorten_count = static_cast<int>((max_shorten - min_shorten) / global_map.resolution) + 1;
        int shorten_i;  double shorten_dis;
        for (
            shorten_i = shorten_count, shorten_dis = min_shorten + (max_shorten - min_shorten) * shorten_i / shorten_count;
            shorten_i >= 1;
            shorten_i -= 1, shorten_dis = min_shorten + (max_shorten - min_shorten) * shorten_i / shorten_count
        ) {
            // 圆与前一段路径的切点
            arc_front = Eigen::Translation2d(Eigen::Translation2d(turn_point) * Eigen::Rotation2Dd(line_front_angle) * Eigen::Vector2d(-shorten_dis, 0)) * Eigen::Rotation2Dd(line_front_angle);
            // 圆与后一段路径的切点
            arc_back = Eigen::Translation2d(Eigen::Translation2d(turn_point) * Eigen::Rotation2Dd(line_back_angle) * Eigen::Vector2d(shorten_dis, 0)) * Eigen::Rotation2Dd(line_back_angle);
            // 检查是否为有效的圆弧
            if (is_curve_passable(arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote, cls_point)) {
                // 无碰撞, 为有效的圆弧
                return true;
            }
        }
        return false;
    }
    
    // [有头模式] 根据简化后的路径, 采用圆弧样条, 生成带有曲率以及距离的轨迹
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> generate_arc_trajectory(
        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> path
    ) {
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> opti_path;
        opti_path.emplace_back(path[0], 0, 0);

        if (path.size() >= 3) {
            // 仅当路径点数 >= 3时进行

            // 计算每一个路径的累积距离
            std::vector<double> length(path.size(), 0);
            for (size_t i = 1; i < path.size(); i++) {
                length[i] = length[i - 1] + (path[i - 1].translation() - path[i].translation()).norm();
            }

            // 计算每一段线段的角度
            for (size_t i = 0; i < path.size() - 1; i++) {
                Eigen::Vector2d diff_pos = path[i + 1].translation() - path[i].translation();
                double line_angle = std::atan2(diff_pos[1], diff_pos[0]);
                path[i] = Eigen::Translation2d(path[i].translation()) * Eigen::Rotation2Dd(line_angle);
            }

            //for (const Eigen::Affine2d& p : path) {
            //    opti_path.emplace_back(p, 0, 0);
            //}

            // 合并相邻的若干个拐点, 角度变化应 < 179°, 角度变化方向应相同
            int merge_front = 1;
            while (merge_front < (int)path.size() - 1) {
                // 贪心算法, 一次性合并最多的点

                double front_diff_angle = hust::diff_angle(path[merge_front - 1], path[merge_front]);
                // 在不考虑地图的情况下, 计算一次性最多合并多少个拐点
                int max_merge_back = merge_front;
                for (int merge_back = merge_front + 1; merge_back < (int)path.size() - 1; merge_back++) {
                    // 检查角度变化方向是否相同
                    double diff_angle = hust::diff_angle(path[merge_back - 1], path[merge_back]);
                    if (diff_angle * front_diff_angle < 0) {
                        break;
                    }
                    // 检查角度变化量 < 179
                    double sum_diff_angle = hust::diff_angle(path[merge_front - 1], path[merge_back]);
                    if (std::abs(sum_diff_angle) > 179.0 / 180 * M_PI) {
                        break;
                    }
                    max_merge_back = merge_back;
                }

                // 从多到少, 考虑地图的情况下, 尝试尽可能多的合并拐点
                for (int merge_back = max_merge_back; merge_back >= merge_front; merge_back--) {
                    // 计算合并后的拐点 与 最大截短长度

                    // 起始拐点
                    Eigen::Affine2d p0 = Eigen::Translation2d(path[merge_front].translation()) * Eigen::Rotation2Dd(path[merge_front-1].rotation());
                    // 结束拐点
                    Eigen::Affine2d p1 = Eigen::Translation2d(path[merge_back].translation()) * Eigen::Rotation2Dd(hust::get_angle(path[merge_back]) + M_PI);

                    // 使用正弦定理计算新拐点到起始拐点的距离
                    Eigen::Vector2d line = p1.translation() - p0.translation();
                    double line_angle = std::atan2(line[1], line[0]);
                    double beta = std::abs(hust::sub_2pi(line_angle, hust::get_angle(p1)));
                    double gamma = std::abs(hust::diff_angle(p0, p1));
                    double C = line.norm();
                    double B = std::sin(beta) / std::sin(gamma) * C;
                    double p0_to_turn_p = B;

                    // 新拐点的位置
                    Eigen::Vector2d merged_turn_point = p0 * Eigen::Vector2d(p0_to_turn_p, 0);

                    // 计算前后最大截短长度
                    double front_max = (length[merge_front] - length[merge_front - 1]) / 2;
                    double back_max = (length[merge_back + 1] - length[merge_back]) / 2;
                    if (merge_front == 1) {
                        front_max = length[merge_front];
                    }
                    if (merge_back == path.size() - 2) {
                        back_max = length[merge_back + 1] - length[merge_back];
                    }
                    front_max += (merged_turn_point - p0.translation()).norm();
                    back_max += (merged_turn_point - p1.translation()).norm();
                    double max_shorten = std::min(front_max, back_max);

                    // 计算前后最小截短长度
                    double front_min = (merged_turn_point - p0.translation()).norm();
                    double back_min = (merged_turn_point - p1.translation()).norm();
                    double min_shorten = std::max(front_min, back_min);

                    Eigen::Affine2d arc_front;          // 圆弧起点
                    Eigen::Affine2d arc_back;            // 圆弧终点
                    double arc_radius;                  // 曲率半径
                    Eigen::Vector2d arc_center;         // 圆弧圆心
                    double arc_interplote_interval;     // 圆弧内插间隔
                    std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> arc_interplote; // 圆弧内插点

                    // 检查合并后的拐点是否能够使用圆弧过渡
                    Eigen::Vector2d cls_point;
                    if (turn_has_passable_curve(
                        merged_turn_point,
                        hust::get_angle(path[merge_front - 1]), hust::get_angle(path[merge_back]),
                        max_shorten, min_shorten, 
                        arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote, &cls_point)
                    ) {
                        // 找到了有效圆弧

                        //opti_path.emplace_back(p0, 0, 0);
                        //opti_path.emplace_back(Eigen::Translation2d(merged_turn_point) * Eigen::Rotation2Dd(0), 0, 0);
                        //opti_path.emplace_back(p1, 0, 0);

                        // 插入圆弧头
                        opti_path.emplace_back(
                            arc_front,
                            1 / arc_radius,
                            opti_path.back().length + (opti_path.back().pos.translation() - arc_front.translation()).norm());
                        // 插入圆弧内插点
                        for (const Eigen::Affine2d& p : arc_interplote) {
                            opti_path.emplace_back(
                                p,
                                1 / arc_radius,
                                opti_path.back().length + arc_interplote_interval);
                        }
                        // 插入圆弧终点
                        opti_path.emplace_back(
                            arc_back,
                            0,
                            opti_path.back().length + arc_interplote_interval);

                        merge_front = merge_back + 1;
                        break;
                    }
                    else {
                        if (merge_front == merge_back) {
                            // 始终未找到有效圆弧, 出错
                            temp_cls_point = cls_point;
                            throw std::runtime_error("cannot find avalable arc!!!");
                        }
                    }
                }
            }
            


        //    // 计算每一个拐点的最大截短长度
        //    std::vector<double> max_shorten(path.size());
        //    max_shorten[0] = 0;
        //    max_shorten.back() = 0;
        //    for (size_t i = 1; i < path.size() - 1; i++) {
        //        double front_max = (length[i] - length[i - 1]) / 2;
        //        double back_max = (length[i + 1] - length[i]) / 2;
        //        if (i == 1) {
        //            front_max = length[i];
        //        }
        //        if (i == path.size() - 2) {
        //            back_max = length[i + 1] - length[i];
        //        }
        //        max_shorten[i] = std::min(front_max, back_max);
        //    }

        //    // 根据地图情况, 计算每一个拐点的实际最大截短长度
        //    for (size_t turn_i = 1; turn_i < path.size() - 1; turn_i++) {


        //        Eigen::Affine2d arc_front;          // 圆弧起点
        //        Eigen::Affine2d arc_back;            // 圆弧终点
        //        double arc_radius;                  // 曲率半径
        //        Eigen::Vector2d arc_center;         // 圆弧圆心
        //        double arc_interplote_interval;     // 圆弧内插间隔
        //        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> arc_interplote; // 圆弧内插点

        //        // 检查该拐点是否能够使用圆弧过渡
        //        if (turn_has_valid_curve(
        //            path[turn_i].translation(), 
        //            hust::get_angle(path[turn_i - 1]), hust::get_angle(path[turn_i]),
        //            max_shorten[turn_i], 
        //            arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote)
        //        ) {
        //            // 找到了有效圆弧
        //            // 插入圆弧头
        //            opti_path.emplace_back(
        //                arc_front,
        //                1 / arc_radius,
        //                opti_path.back().length + (opti_path.back().pos.translation() - arc_front.translation()).norm());
        //            // 插入圆弧内插点
        //            for (const Eigen::Affine2d& p : arc_interplote) {
        //                opti_path.emplace_back(
        //                    p,
        //                    1 / arc_radius,
        //                    opti_path.back().length + arc_interplote_interval);
        //            }
        //            // 插入圆弧终点
        //            opti_path.emplace_back(
        //                arc_back,
        //                0,
        //                opti_path.back().length + arc_interplote_interval);
        //        }
        //        else {
        //            // 未找到有效圆弧, 出错
        //            throw std::runtime_error("cannot find avalable arc!!!");
        //        }
        //    }
        }

        // 插入路径终点
        opti_path.emplace_back(path.back(), 0,
            opti_path.back().length + (opti_path.back().pos.translation() - path.back().translation()).norm());

        return opti_path;
    }

    // 计算点到最近的轨迹点
    static bool nearest_traj_point(
        const Eigen::Affine2d pos,
        const std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>>& path,
        int* p_nearest_next = NULL,     // 下一个路径点
        double* p_err_linear = NULL,    // 位置差
        double* p_err_rad = NULL        // 角度差
    ) noexcept
    {
        if (path.empty()) {
            // 路径为空
            return false;
        }

        int nearest_next;
        double err_linear = std::numeric_limits<double>::infinity();
        double err_rad;

        Eigen::Vector2d car_xy = pos.translation();
        double car_angle = Eigen::Rotation2Dd(pos.rotation()).angle();

        if (path.size() == 1) {
            // 轨迹长度为 1
            nearest_next = 0;
            err_linear = (car_xy - path[0].pos.translation()).norm();
            err_rad = hust::diff_angle(pos, path[0].pos);
        }
        else {
            // 轨迹长度 >= 2
            // 获取与当前车辆位置最近的轨迹线段, 并记录位置偏差
            for (int i = path.size() - 1; i > 0; i--) {
                // 前后路径点
                Eigen::Affine2d next = path[i].pos;
                Eigen::Affine2d prev = path[i - 1].pos;
                // 路径点坐标
                Eigen::Vector2d next_xy = next.translation();
                Eigen::Vector2d prev_xy = prev.translation();
                // 路径线段到车身的直线距离
                double line_percent;
                double distance = min_dist_to_line_seg(car_xy, prev_xy, next_xy, NULL, &line_percent);
                if (distance < err_linear) {
                    // 记录位置偏差
                    nearest_next = i;
                    err_linear = distance;
                    // 计算路径线段上的角度差
                    double next_angle = Eigen::Rotation2Dd(path[i].pos.rotation()).angle();
                    double prev_angle = Eigen::Rotation2Dd(path[i - 1].pos.rotation()).angle();
                    double diff_angle = hust::sub_2pi(prev_angle, next_angle);
                    // 计算车辆角度与轨迹的差
                    err_rad = hust::sub_2pi(car_angle, (prev_angle + diff_angle * line_percent));
                }
            }
        }

        if (p_nearest_next) {
            *p_nearest_next = nearest_next;
        }
        if (p_err_linear) {
            *p_err_linear = err_linear;
        }
        if (p_err_rad) {
            *p_err_rad = err_rad;
        }
        return true;
    }

    // [有头模式] 内插轨迹, 仅在曲率为0或NaN的线段上进行内插
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> interplote_trajectory(
        const std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>>& origin_traj
    ) {
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> new_traj;
        for (int i = 0; i < (int)origin_traj.size() - 1; i++) {
            const HeadTrajectoryPoint& p0 = origin_traj[i];
            const HeadTrajectoryPoint& p1 = origin_traj[i+1];
            Eigen::Vector2d line = p1.pos.translation() - p0.pos.translation();
            int interplote_count = static_cast<int>(std::ceil(line.norm() / global_map.resolution)) - 1;
            new_traj.push_back(p0);
            if ((std::isnan(p0.curvature) || p0.curvature == 0) && interplote_count >= 1) {
                // 需要内插
                double p0_rad = hust::get_angle(p0.pos);
                double p1_rad = hust::get_angle(p1.pos);
                double diff_rad = hust::sub_2pi(p0_rad, p1_rad);
                for (int i = 0; i < interplote_count; i++) {
                    double percent = 1.0 * (i + 1) / (interplote_count + 1);
                    new_traj.emplace_back(
                        Eigen::Translation2d(p0.pos.translation() + line * percent) *
                        Eigen::Rotation2Dd(p0_rad + diff_rad * percent),
                        p0.curvature,                                   // 曲率跟随起点
                        p0.length + percent * line.norm()                           // 长度
                    );
                }
            }
        }
        new_traj.push_back(origin_traj.back());
        return new_traj;
    }


    // [有头模式] 根据动力学约束, 时间参数化带有正确曲率与距离信息的轨迹
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> time_arg_trajectory(
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> traj, 
        Eigen::Vector2d begin_vel_xy, // 当前平移速度
        double begin_vel_rad,     // 当前角速度
        double max_accel_linear,// 最大线加速度
        double max_accel_rad,   // 最大角加速度
        double max_vel_linear,
        double max_vel_rad,
        double tol_linear = 0.005,
        double tol_rad = 0.1 / 180 * M_PI
    ) {
        // 根据最大向心加速度与路径曲率, 计算最大速度
        std::vector<double> max_speed(traj.size());
		std::vector<double> max_speed_rad(traj.size());
        // 出发点速度为当前速度在运动方向上的投影, 不为负
        double begin_speed = begin_vel_xy.dot(traj[0].pos.rotation() * Eigen::Vector2d(1, 0));
        if (begin_speed < 0) {
            begin_speed = 0;
        }
        max_speed.front() = begin_speed;
		max_speed_rad.front() = begin_vel_rad;
        // 末端速度为0
		max_speed.back() = 0;
		max_speed_rad.back() = 0;
		for (int i = 1; i < (int)traj.size() - 1; i++) {
            const auto& p = traj[i];
            if (p.curvature == 0 || std::isnan(p.curvature)) {
                // 曲率为0或未知, 速度无限制
                max_speed[i] = std::numeric_limits<double>::infinity();
            }
            else if (std::isinf(p.curvature)) {
                // 曲率无限大, 停车自转
                max_speed[i] = 0;
            }
            else {
                // 曲率有效
                double radius = std::abs(1 / p.curvature);  // 曲率半径
                max_speed[i] = std::sqrt(max_accel_linear * radius);    // 限制向心加速度sqrt(a*R)
            }
			max_speed_rad[i] = std::numeric_limits<double>::infinity();
        }

        // todo: 根据角速度动力学限制, 计算最大角速度

        // 限制最大速度不超过给定值
        for (double& v : max_speed) {
            if (v > max_vel_linear) {
                v = max_vel_linear;
            }
        }
		for (double& w : max_speed_rad) {
			if (w > max_vel_rad) {
				w = max_vel_rad;
			}
		}

        // 根据加速度限制, 向前传播, 计算每个轨迹点的速度限制
        for (int i = traj.size() - 2; i >= 0; i--) {
            // 平移距离
            double distance = traj[i + 1].length - traj[i].length;
            // 下一点的线速度
            double vel_next = max_speed[i + 1];
            // 加速度设为最大加速度
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_prev = std::sqrt(2 * max_accel_linear * distance + vel_next * vel_next);
            if (max_speed[i] > vel_prev) {
                max_speed[i] = vel_prev;
            }
        }
        // 根据加速度限制, 向后传播
        for (size_t i = 1; i < traj.size(); i++) {
            // 平移距离
            double distance = traj[i].length - traj[i - 1].length;
            // 上一点的线速度
            double vel_prev = max_speed[i - 1];
            // 加速度设为最大加速度
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_next = std::sqrt(2 * max_accel_linear * distance + vel_prev * vel_prev);
            if (max_speed[i] > vel_next) {
                max_speed[i] = vel_next;
            }
			// 计算角度变化量
			double diff_angle = hust::diff_angle(traj[i].pos, traj[i - 1].pos);
			// 上一点的角速度
			double rad_prev = max_speed_rad[i - 1];
			// 以最大角速度计算所需时间
			double rad_next = std::sqrt(2 * max_accel_rad * std::abs(diff_angle) + rad_prev * rad_prev);
			if (max_speed_rad[i] > rad_next) {
				max_speed_rad[i] = rad_next;
			}
        }

        // 根据期望速度以及距离, 计算每一点的预计时间
        traj[0].time = 0;
        for (size_t i = 1; i < traj.size(); i++) {
            double avg_vel = (max_speed[i] + max_speed[i-1]) / 2;
            double distance = traj[i].length - traj[i - 1].length;
			double dt_linear = distance / avg_vel;
			if (distance == 0 && avg_vel == 0) {
				dt_linear = 0;
			}

			double avg_rad = (max_speed_rad[i] + max_speed_rad[i - 1]) / 2;
			double diff_angle = hust::diff_angle(traj[i].pos, traj[i - 1].pos);
			double dt_rad = std::abs(diff_angle) / std::abs(avg_rad);
			if (diff_angle == 0 && avg_rad == 0) {
				dt_rad = 0;
			}

			// 取两个自由度中较慢的时间
			if (dt_linear > dt_rad) {
				traj[i].time = traj[i - 1].time + dt_rad;
			}
			else {
				traj[i].time = traj[i - 1].time + dt_linear;
			}
        }

        // 根据时间与角度差, 计算速度与角速度前馈
        for (size_t i = 0; i < traj.size() - 1; i++) {
			double duration = traj[i + 1].time - traj[i].time;

			double distance = traj[i + 1].length - traj[i].length;
			double vel_linear = distance / duration;
			if (distance == 0 && duration == 0) {
				vel_linear = 0;
			}
			else if (distance > 0 && duration == 0) {
				vel_linear = max_vel_linear;
			}
			else if (distance < 0 && duration == 0) {
				vel_linear = -max_vel_linear;
			}

            double diff_angle = hust::diff_angle(traj[i].pos, traj[i + 1].pos);
			double vel_rad = diff_angle / duration;
			if (diff_angle == 0 && duration == 0) {
				vel_rad = 0;
			}
			else if (diff_angle > 0 && duration == 0) {
				vel_rad = max_vel_rad;
			}
			else if (diff_angle < 0 && duration == 0) {
				vel_rad = -max_vel_rad;
			}

			traj[i].vel_linear = traj[i].pos.rotation() * Eigen::Vector2d(max_speed[i], 0);
			traj[i].vel_rad = vel_rad;
        }
		traj.back().vel_linear = Eigen::Vector2d(0, 0);
		traj.back().vel_rad = 0;
		return traj;
    }

    // [废弃的] 内插方式
    enum class PathInterplateType {
        SAME_DISTANCE,      // 等距
        ALIGNED_TO_GRID,    // 对齐到X或Y坐标
    };

    // [废弃的] 在轨迹上进行内插
    static std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> interplate_path(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        double resolution,      // 内插间隔
        PathInterplateType type = PathInterplateType::SAME_DISTANCE // 内插类型
    ) {
        if (path.size() < 2) {
            return path;
        }

        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> new_path;
        if (type == PathInterplateType::SAME_DISTANCE) {
            // 等距内插
            for (size_t i = 0; i < path.size() - 1; i++) {
                new_path.push_back(path[i]);
                double distance = (path[i].translation() - path[i + 1].translation()).norm();
                int count = static_cast<int>(std::ceil(distance / resolution));
                if (count > 1) {
                    // 需要内插
                    Eigen::Vector2d p0_xy = path[i].translation();
                    Eigen::Vector2d p1_xy = path[i+1].translation();
                    Eigen::Vector2d diff_xy = p1_xy - p0_xy;
                    double p0_rad = Eigen::Rotation2Dd(path[i].rotation()).angle();
                    double p1_rad = Eigen::Rotation2Dd(path[i+1].rotation()).angle();
                    double diff_rad = hust::sub_2pi(p0_rad, p1_rad);
                    for (int i = 1; i < count; i++) {
                        new_path.push_back(
                            Eigen::Translation2d(p0_xy + diff_xy * i / count) *
                            Eigen::Rotation2Dd(p0_rad + diff_rad * i / count));
                    }
                }
            }
            new_path.push_back(path.back());
        }
        else {
            // 对齐到X或Y坐标内插
            // todo
        }
        return new_path;
    }

    // [废弃的] 根据内插后的路径生成带有速度的轨迹
    static std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> generate_trajectory(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path, 
        Eigen::Vector2d cur_vel_xy, // 当前平移速度
        double cur_vel_rad,     // 当前角速度
        double max_accel_linear,// 最大线加速度
        double max_accel_rad,   // 最大角加速度
        double max_vel_linear,
        double max_vel_rad,
        double tol_linear = 0.005,
        double tol_rad = 0.1 / 180 * M_PI)
    {
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> traj;

        if (path.empty()) {
            return traj;
        }

        traj.reserve(path.size());
        for (const Eigen::Affine2d& p : path) {
            HeadTrajectoryPoint t;
            t.pos = p;
            traj.push_back(t);
        }

        // 求路径长度
        double total_length = 0;
        traj[0].length = 0;
        for (size_t i = 1; i < traj.size(); i++) {
            total_length += (traj[i - 1].pos.translation() - traj[i].pos.translation()).norm();
            traj[i].length = total_length;
        }

        // 有拐弯则按最最小转弯半径 0.2 计算曲率与线速度方向
        double turn_curvature = 1 / 0.2;
        traj[0].curvature = 0;
        // 速度为投影
        traj[0].vel_linear = (traj[1].pos.translation() - traj[0].pos.translation()).normalized() * ((traj[1].pos.translation() - traj[0].pos.translation()).normalized().dot(cur_vel_xy));
        for (size_t i = 1; i < traj.size() - 1; i++) {
            // 前后三个点
            Eigen::Vector2d p0 = traj[i - 1].pos.translation();
            Eigen::Vector2d p1 = traj[i].pos.translation();
            Eigen::Vector2d p2 = traj[i + 1].pos.translation();
            // 两个线段
            Eigen::Vector2d l_01 = p1 - p0;
            Eigen::Vector2d l_12 = p2 - p1;
            // 检查线段长度
            if (l_01.norm() < 0.005 || l_12.norm() < 0.005) {
                // 线段过短
                traj[i].curvature = turn_curvature;
            }
            else {
                // 两个线段的角度
                double angle_line_01 = std::atan2(l_01[1], l_01[0]);
                double angle_line_12 = std::atan2(l_12[1], l_12[0]);
                // 两个线段的夹角
                double diff_angle = hust::sub_2pi(angle_line_01, angle_line_12);
                if (std::abs(diff_angle) > 50 * M_PI / 180) {
                    // 急转弯, 需要停车
                    if (diff_angle > 0) {
                        traj[i].curvature = std::numeric_limits<double>::infinity();
                    }
                    else {
                        traj[i].curvature = -std::numeric_limits<double>::infinity();
                    }
                    traj[i].vel_linear = Eigen::Vector2d(0, 0);
                }
                else if (std::abs(diff_angle) > 0.1 * M_PI / 180) 
                {
                    // 有拐弯
                    traj[i].curvature = turn_curvature;
                    traj[i].vel_linear = l_12.normalized() * 1.5;
                }
                else {
                    // 直线
                    traj[i].curvature = 0;
                    traj[i].vel_linear = l_12.normalized() * max_vel_linear;
                }
            }
        }
        // 最后一点速度为0
        traj.back().curvature = 0;
        traj.back().vel_linear = Eigen::Vector2d(0, 0);

        // 根据加速度限制, 向前传播, 计算每个轨迹点的速度限制
        for (int i = traj.size() - 2; i >= 0; i--) {
            // 平移距离
            double distance = traj[i + 1].length - traj[i].length;
            // 下一点的线速度
            double vel_next = traj[i+1].vel_linear.norm();
            // 加速度设为最大加速度
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_prev = std::sqrt(2 * max_accel_linear * distance + vel_next * vel_next);
            if (traj[i].vel_linear.norm() > vel_prev) {
                traj[i].vel_linear = traj[i].vel_linear.normalized() * vel_prev;
            }
        }
        // 根据加速度限制, 向后传播
        for (size_t i = 1; i < traj.size(); i++) {
            // 平移距离
            double distance = traj[i].length - traj[i-1].length;
            // 上一点的线速度
            double vel_prev = traj[i-1].vel_linear.norm();
            // 加速度设为最大加速度
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_next = std::sqrt(2 * max_accel_linear * distance + vel_prev * vel_prev);
            if (traj[i].vel_linear.norm() > vel_next) {
                traj[i].vel_linear = traj[i].vel_linear.normalized() * vel_next;
            }
        }

        // 根据期望速度以及距离, 计算每一点的预计时间
        traj[0].time = 0;
        for (size_t i = 1; i < traj.size(); i++) {
            double avg_vel = (traj[i].vel_linear.norm() + traj[i - 1].vel_linear.norm()) / 2;
            double distance = traj[i].length - traj[i - 1].length;
            traj[i].time = traj[i - 1].time + distance / avg_vel;
        }

        // 根据时间与角度差, 计算角速度前馈
        for (size_t i = 0; i < traj.size() - 1; i++) {
            double diff_angle = hust::diff_angle(traj[i].pos, traj[i + 1].pos);
            double duration = traj[i+1].time - traj[i].time;
            traj[i].vel_rad = diff_angle / duration;
            if (std::isnan(traj[i].vel_rad)) {
                traj[i].vel_rad = 0;
            }
            if (traj[i].vel_rad > max_vel_rad) {
                traj[i].vel_rad = max_vel_rad;
            }
            else if (traj[i].vel_rad < -max_vel_rad) {
                traj[i].vel_rad = -max_vel_rad;
            }
        }
        traj.back().vel_rad = 0;

        return traj;
    }

    // 计算点到线段的最小距离
    static double min_dist_to_line_seg(
        const Eigen::Vector2d& p, 
        const Eigen::Vector2d& v, const Eigen::Vector2d& w, 
        Eigen::Vector2d* nearest = NULL, 
        double* percent=NULL) noexcept
    {
        // Return minimum distance between line segment vw and point p
        double l2 = (w - v).squaredNorm();  // i.e. |w-v|^2 -  avoid a sqrt
        if (l2 == 0.0) {
            if (nearest) {
                *nearest = v;
            }
            if (percent) {
                *percent = 0;
            }
            return (p - v).norm();   // v == w case
        }
                                                // Consider the line extending the segment, parameterized as v + t (w - v).
                                                // We find projection of point p onto the line. 
                                                // It falls where t = [(p-v) . (w-v)] / |w-v|^2
                                                // We clamp t from [0,1] to handle points outside the segment vw.
        double t = std::max(0.0, std::min(1.0, (p - v).dot(w - v) / l2));
        Eigen::Vector2d projection = v + t * (w - v);  // Projection falls on the segment
        if (nearest) {
            *nearest = projection;
        }
        if (percent) {
            *percent = t;
        }
        return (p - projection).norm();
    }

    // 获取默认的车体轮廓点
    static std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> get_default_car() {
        static std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> default_car;
        // 车体默认初始化为 0.6m 宽 0.8m 长
        double r_x = 0.4;
        double r_y = 0.3;
        // 左前
        default_car.emplace_back(r_x, r_y);
        // 左后
        default_car.emplace_back(r_x, -r_y);
        // 右后
        default_car.emplace_back(-r_x, -r_y);
        // 右前
        default_car.emplace_back(-r_x, r_y);
        return default_car;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 将弧度表示的角度, 转换到角度编号
    int convert_angle_to_id(double rad) {
        rad /= (2 * M_PI / angle_number);
        int id = static_cast<int>(std::round(rad));
        while (id < 0) {
            id += angle_number;
        }
        while (id >= angle_number) {
            id -= angle_number;
        }
        return id;
    }

    // 将角度编号转换到弧度表示的角度
    double convert_id_to_angle(int id) {
        double rad = (2 * M_PI / angle_number) * id;
        while (rad < -M_PI) {
            rad += 2 * M_PI;
        }
        while (rad >= M_PI) {
            rad -= 2 * M_PI;
        }
        return rad;
    }

    // 将世界坐标转换到离散状态空间坐标
    std::array<int, 3> convert_world_to_statepos(const Eigen::Affine2d &world_pos) {
        std::array<int, 3> state_pos;
        // 坐标
        Eigen::Vector2i xy = global_map.convert_to_map_pos(world_pos.translation());
        state_pos[0] = xy[0];
        state_pos[1] = xy[1];
        // 旋转
        double theta = Eigen::Rotation2Dd(world_pos.rotation()).angle();
        state_pos[2] = convert_angle_to_id(theta);
        return state_pos;
    }

    // 将离散状态空间坐标转换到世界坐标
    Eigen::Affine2d convert_statepos_to_world(const std::array<int, 3> &state_pos) {
        Eigen::Vector2d xy = global_map.convert_to_world_pos(Eigen::Vector2i(state_pos[0], state_pos[1]));
        return Eigen::Translation2d(xy) * Eigen::Rotation2Dd(convert_id_to_angle(state_pos[2]));
    }

    Planner(
        GlobalMap &_global_map,
        const std::string& motion_prim_path,
        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& car = get_default_car(), 
        double vel_linear = 2,  // 规划中用于计算平移代价
        double vel_rad = 2 * M_PI   // 规划中用于计算旋转代价
    ) : 
        global_map(_global_map), 
        car(car)
    {

        // 计算车体最大半径
        double addition_ratio = 1.0;
        for (const Eigen::Vector2d& p : car) {
            car_perimeter.emplace_back(p[0] * addition_ratio, p[1] * addition_ratio);

            double radius = p.norm() * addition_ratio;
            if (car_max_radius == 0 || radius > car_max_radius) {
                car_max_radius = radius;
            }
        }
        // 计算车体最小半径
        for (size_t i = 0; i < car.size(); i++) {
            double radius = min_dist_to_line_seg({ 0, 0 }, car[i], car[(i + 1) % car.size()]);
            if (car_min_radius == 0 || radius < car_min_radius) {
                car_min_radius = radius;
            }
        }

        // 为了更顺滑, 最大半径设小一些
        //car_max_radius = car_min_radius + 0.06;

        // 获取导航地图
        global_map.set_plan_map_params(car_min_radius, car_max_radius);
        auto id_map = global_map.generate_plan_map();
        last_plan_map_id = id_map.first;
        map = 0xFF - id_map.second;
        map.setTo(0xFF - OccupyMap::PIXEL_FREE, map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));

        // 初始化规划环境
        env = new EnvironmentNAVXYTHETALAT();

        // 像素值 >= 0xFF - PIXEL_NOT_PLAN 的区域为不可规划区域, 因为在该区域内机器人肯定会碰到障碍物
        if (!env->SetEnvParameter("cost_inscribed_thresh", 0xFF - OccupyMap::PIXEL_NOT_PLAN)) {
            throw SBPL_Exception("ERROR: failed to set parameters");
        }
        // 像素值 >= 0xFF - PIXEL_MAY_CLS 的区域内机器人可能会碰到障碍物
        if (!env->SetEnvParameter("cost_possibly_circumscribed_thresh", 0xFF - OccupyMap::PIXEL_MAY_CLS)) {
            throw SBPL_Exception("ERROR: failed to set parameters");
        }

        // 地图中像素值 >= 0xFF - PIXEL_LIMITED, 表示为障碍物
        if (!env->InitializeEnv(
            map.cols, map.rows, map.data,
            // 起点
            grid_start[0] * global_map.resolution,
            grid_start[1] * global_map.resolution,
            grid_start[2] * (2 * M_PI / angle_number), 
            // 终点
            grid_goal[0] * global_map.resolution,
            grid_goal[1] * global_map.resolution,
            grid_goal[2] * (2 * M_PI / angle_number), 
            // 终点容差
            0.001, 0.001, 0.001, 
            car_perimeter, global_map.resolution,
            1 / vel_linear, M_PI / 4 / vel_rad, 0xFF - OccupyMap::PIXEL_LIMITED,
            motion_prim_path.c_str()
        )) {
            throw SBPL_Exception("ERROR: failed to InitializeEnv");
        }

        // initialize MDP info
        MDPConfig MDPCfg;
        if (!env->InitializeMDPCfg(&MDPCfg)) {
            throw SBPL_Exception("ERROR: InitializeMDPCfg failed");
        }

        // 创建一个规划器 ARA规划器, 正向搜索
        planner = new ARAPlanner(env, true);
        // 设定规划器的开始和结束状态
        if (planner->set_start(MDPCfg.startstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set start state");
        }
        if (planner->set_goal(MDPCfg.goalstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set goal state");
        }
        // 设定初始解的最差限度为最优解的n倍
        planner->set_initialsolution_eps(1.2);
        // 无需找到初始解才返回, 超时就返回
        planner->set_search_mode(false);
    }

    ~Planner() {
        if (plan_thread.joinable()) {
            plan_thread.join();
        }

        delete planner;
        delete env;
    }

    // 通知规划器 起点发生了变化
    bool set_start(const Eigen::Affine2d& start, std::array<int, 3> &new_start = std::array<int, 3>{ {0, 0, 0}}) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        // 转换到离散状态空间
        new_start = convert_world_to_statepos(start);
        if (buffer_grid_start != new_start) {
            buffer_grid_start = new_start;
            buffer_accurate_start = start;
            return true;
        }
        else {
            buffer_accurate_start = start;
            accurate_start = start;
            return false;
        }
    }
    // 通知规划器 目标发生了变化
    bool set_goal(const Eigen::Affine2d& goal, std::array<int, 3> &new_goal = std::array<int, 3>{ { 0, 0, 0 } }) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        // 转换到离散状态空间
        new_goal = convert_world_to_statepos(goal);
        if (buffer_grid_goal != new_goal) {
            buffer_grid_goal = new_goal;
            buffer_accurate_goal = goal;
            // 若终点发生了更新, 清空此前的有效轨迹缓冲
            last_available_trajectory.clear();
            return true;
        }
        else {
            buffer_accurate_goal = goal;
            accurate_goal = goal;
            return false;
        }
    }

    std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> get_path() {
        std::lock_guard<std::mutex> lk(mutex_buffer);
        return solution_backend;
    }

    // 根据给定起点, 当前速度与 终点, 尝试获取规划的地图
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> get_trajectory(
        const Eigen::Affine2d& car_pos,
        const Eigen::Affine2d& target_pos, 
        const std::array<double, 3>& car_speed,
        double max_accel_linear, double max_accel_rad,
        double max_vel_linear, double max_vel_rad
    ) {
        int nearest_traj_i;
        double err_linear;
        double err_rad;

        // 更新规划起点
        set_start(car_pos);

        // 更新规划终点
        set_goal(target_pos);

        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> trajectory;

        err_linear = (car_pos.translation() - target_pos.translation()).norm();
        err_rad = hust::diff_angle(car_pos, target_pos);
        if (err_linear < 0.05) {
            // 起点接近终点, 直接过去
            trajectory.push_back(HeadTrajectoryPoint(car_pos));
            trajectory.push_back(HeadTrajectoryPoint(target_pos));
            return trajectory;
        }

        mutex_buffer.lock();
        if (!last_available_trajectory.empty() && global_map.is_newest_plan_map(last_check_map_id)) {
            // 存在有效轨迹缓存, 直接使用缓存轨迹
            trajectory = last_available_trajectory;
            mutex_buffer.unlock();
        }
        else {
            mutex_buffer.unlock();

            // 没有有效轨迹缓存, 需要重新生成轨迹
            if (!global_map.is_newest_plan_map(last_check_map_id)) {
                // 当前导航地图过期
                // 更新地图
                global_map.set_plan_map_params(car_min_radius, car_max_radius);
                auto id_map = global_map.generate_plan_map();
                last_check_map_id = id_map.first;

                cv::Mat new_map(0xFF - id_map.second);
                new_map.setTo(0xFF - OccupyMap::PIXEL_FREE, new_map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));
                new_map.copyTo(map);
                hust::log("[PLAN] Re-generate plan map to check trajectory, id=" + std::to_string(id_map.first));
            }

            // 获取最新规划的路径
            mutex_buffer.lock();
            auto path = solution_backend;
            mutex_buffer.unlock();

            if (path.empty()) {
                // 无解
                hust::log_status("[PLAN] Empty solution path");
                trajectory = {};
                last_available_trajectory = {};
                return trajectory;
            }

            if (target_pos.translation()[0] < -2) {
                hust::log_status("[PLAN] task3 target");
            }

            // 检查路径到目标位置的差
            if (false) {
                if (nearest_path_point(target_pos, path, &nearest_traj_i, &err_linear, &err_rad)) {
                    if (nearest_traj_i < 1) {
                        // 已知路径, 无法到达目标点, 无解
                        hust::log_status("[PLAN] path cannot reach goal (nearest_traj_i < 1)");
                        trajectory = {};
                        last_available_trajectory = {};
                        return trajectory;
                    }
                    else if (err_linear > 0.1 || std::abs(err_rad) > M_PI / 4) {
                        // 检查轨迹位置到目标位置否可达
                        // todo: 考虑上角度变化
                        if (!is_line_free(target_pos.translation(), path[nearest_traj_i - 1].translation())) {
                            // 不可达, 无解
                            hust::log_status("[PLAN] path cannot reach goal (occupied)");
                            trajectory = {};
                            last_available_trajectory = {};
                            return trajectory;
                        }
                    }
                }
                else {
                    // 不应该出现该种情况
                    throw std::runtime_error("cannot find nearest_traj_point to target_pos");
                }

                // 删除多余的路径点
                path.erase(path.begin() + nearest_traj_i, path.end());
                // 将目标点作为路径终点
                path.push_back(target_pos);
            }
            else {
                err_linear = (target_pos.translation() - path.back().translation()).norm();
                err_rad = hust::diff_angle(target_pos, path.back()); 
                if (err_linear > 0.1 || std::abs(err_rad) > M_PI / 4) {
                    // 检查轨迹位置到目标位置否可达
                    // todo: 考虑上角度变化
                    if (!is_line_free(target_pos.translation(), path.back().translation())) {
                        // 不可达, 无解
                        hust::log_status("[PLAN] path cannot reach goal (occupied)");
                        trajectory = {};
                        last_available_trajectory = {};
                        return trajectory;
                    }
                }
            }

            for (int n = 10; n >= 0; n--) {
                safe_percent = min_safe_percent_generate + n / 10.0 * (1 - min_safe_percent_generate);

                // 生成轨迹, 若由于路径非法, 无法生成有效轨迹, 则会抛出异常
                try {
                    trajectory = generate_arc_trajectory(path); // 耗时!!!
                    temp_trajectory = trajectory;
                    break;
                }
                catch (const std::exception& err) {
                    if (n == 0) {
                        // 路径被占据, 无解
                        hust::log_status("[PLAN] arc path is occupied (" + std::string(err.what()) + ")");
                        trajectory = {};
                        last_available_trajectory = {};
                        return trajectory;
                    }
                    else {
                        continue;
                    }
                }
            }

            for (int n = 10; n >= 0; n--) {
                safe_percent = min_safe_percent_test + n / 10.0 * (1 - min_safe_percent_test);

                // 检查轨迹是否可行
                bool is_occupied = false;
                for (size_t i = 0; i < trajectory.size() - 1; i++) {
                    Eigen::Vector2d cls_point;
                    if (!is_line_passable(trajectory[i].pos, trajectory[i + 1].pos, false, &cls_point)) {
                        if (n == 0) {
                            // 路径被占据, 无解
                            temp_cls_point = cls_point;

                            hust::log_status("[PLAN] path is occupied");
                            trajectory = {};
                            last_available_trajectory = {};
                            return trajectory;
                        }
                        else {
                            is_occupied = true;
                        }
                    }
                }
                if (!is_occupied) {
                    break;
                }
            }

            // 更新有效轨迹缓存
            hust::log("[PLAN] Last trajectory is ok id=" + std::to_string(last_check_map_id));
            mutex_buffer.lock();
            last_available_trajectory = trajectory;
            mutex_buffer.unlock();
        }

        // 在轨迹中的直线线段上进行内插
        trajectory = interplote_trajectory(trajectory);

        // 获得距离当前位置最近的轨迹点
        if (nearest_traj_point(car_pos, trajectory, &nearest_traj_i, &err_linear, &err_rad)) {
            // 删除过期路径
            double passed_length = trajectory[nearest_traj_i].length;
            trajectory.erase(trajectory.begin(), trajectory.begin() + nearest_traj_i);
            for (HeadTrajectoryPoint& p : trajectory) {
                p.length -= passed_length;
            }
        }
        else {
            // 不应该出现该种情况
            throw std::runtime_error("cannot find nearest_traj_point to car_pos");
        }

        // 检查当前位置到轨迹起点是否可达
        // todo: 考虑上角度变化
        if (!is_line_passable(car_pos, trajectory.front().pos) && is_pos_passable(car_pos)) {
            // 不可达, 无解
            hust::log_status("[PLAN] car cannot reach traj start");
            trajectory = {};
            last_available_trajectory = {};
            return trajectory;
        }

        // 插入当前位置作为真实起点
        double cur_to_front = (car_pos.translation() - trajectory.front().pos.translation()).norm();
        for (HeadTrajectoryPoint& p : trajectory) {
            p.length += cur_to_front;
        }
        trajectory.insert(
            trajectory.begin(),
            HeadTrajectoryPoint(car_pos, std::numeric_limits<double>::signaling_NaN(), 0)
        );

        // 还原终点位姿
        trajectory.back().pos = Eigen::Translation2d(trajectory.back().pos.translation()) * Eigen::Rotation2Dd(target_pos.rotation());

        // 时间参数化轨迹
        trajectory = time_arg_trajectory(trajectory,
            Eigen::Vector2d(car_speed[0], car_speed[1]), car_speed[2],
            max_accel_linear, max_accel_rad,
            max_vel_linear, max_vel_rad);

        hust::log_status("[PLAN] OK");
        temp_trajectory = trajectory;
        return trajectory;
    }

    // 内部规划线程
    void _thread_plan() {
        std::vector<int> solution_state;
        while (true) {

            mutex_buffer.lock();
            // 检查是否退出
            if (stop_plan) {
                stop_plan = true;
                mutex_buffer.unlock();
                break;
            }
            // 起始位置有更新
            bool start_updated = false;
            bool goal_updated = false;
            if (grid_start != buffer_grid_start) {
                if (env->SetStart(
                    buffer_grid_start[0] * global_map.resolution,
                    buffer_grid_start[1] * global_map.resolution,
                    buffer_grid_start[2] * (2 * M_PI / angle_number)
                ) == -1) {
                    throw SBPL_Exception("ERROR: SetStart failed");
                }
                grid_start = buffer_grid_start;
                accurate_start = buffer_accurate_start;
                start_updated = true;
            }
            // 目标位置有更新
            if (grid_goal != buffer_grid_goal) {
                if (env->SetGoal(
                    buffer_grid_goal[0] * global_map.resolution,
                    buffer_grid_goal[1] * global_map.resolution,
                    buffer_grid_goal[2] * (2 * M_PI / angle_number)
                ) == -1) {
                    throw SBPL_Exception("ERROR: SetGoal failed");
                }
                grid_goal = buffer_grid_goal;
                accurate_goal = buffer_accurate_goal;
                goal_updated = true;
            }
            mutex_buffer.unlock();

            if (start_updated || goal_updated) {
                // 起始或目标位置发生变更后, 更新规划器的属性
                MDPConfig MDPCfg;
                if (!env->InitializeMDPCfg(&MDPCfg)) {
                    throw SBPL_Exception("ERROR: InitializeMDPCfg failed");
                }
                if (start_updated) {
                    if (planner->set_start(MDPCfg.startstateid) == 0) {
                        throw SBPL_Exception("ERROR: failed to set start state");
                    }
                }
                if (goal_updated) {
                    if (planner->set_goal(MDPCfg.goalstateid) == 0) {
                        throw SBPL_Exception("ERROR: failed to set goal state");
                    }
                }
            }

            // 检查规划地图是否有更新
            global_map.set_plan_map_params(car_min_radius, car_max_radius);
            auto id_map = global_map.generate_plan_map();
            int id = id_map.first;
            mutex_buffer.lock();
            if (last_plan_map_id != id) {
                // 导航地图有变化
                last_plan_map_id = id;
                mutex_buffer.unlock();

                cv::Mat new_map(0xFF - id_map.second);
                new_map.setTo(0xFF - OccupyMap::PIXEL_FREE, new_map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));
                new_map.copyTo(map);
                // ARA规划器不支持增量的地图更新
                env->SetMap(map.data);
                ((ARAPlanner*)planner)->costs_changed();
                hust::log("[PLAN] Fully replan because map changed!!! id=" + std::to_string(last_plan_map_id));
            }
            else {
                mutex_buffer.unlock();
            }

            // 进行规划
            int ret = planner->replan(0.1, &solution_state);
            if (ret == 1) {
                // 成功找到解
                // 将状态空间的解转换到X Y Theta空间
                std::vector<sbpl_xy_theta_pt_t> path;
                env->ConvertStateIDPathintoXYThetaPath(&solution_state, &path);

                std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> solution;

                // 输出规划结果
                for (const sbpl_xy_theta_pt_t &p : path) {
                    // 转换到世界坐标系 Eigen::Affine2d
                    solution.emplace_back(
                        Eigen::Translation2d(p.x + global_map.left_bottom[0], p.y + global_map.left_bottom[1]) * 
                        Eigen::Rotation2Dd(p.theta));
                }

                // 简化规划结果
                solution = simplify_path(solution, true);

                mutex_buffer.lock();
                solution_backend = solution;
                mutex_buffer.unlock();
                cv_has_solution.notify_all();
            }
            else {
                // 无解
                //mutex_buffer.lock();
                //solution_backend.clear();
                //mutex_buffer.unlock();
            }
        }
    }

    // 开始在新线程中进行规划
    void start_plannig() {
        if (!plan_thread.joinable()) {
            plan_thread = std::thread(std::bind(&Planner::_thread_plan, this));
        }
    }

    // 等待获取解
    void wait_for_solution() {
        std::unique_lock<std::mutex> lk(mutex_buffer);
        cv_has_solution.wait(lk, [this] {return solution_backend.size() > 0; });
    }

    // 结束规划线程
    void stop_planning(bool wait_for_stop = false) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        stop_plan = true;
        // 等待线程结束
        if (wait_for_stop) {
            if (plan_thread.joinable()) {
                plan_thread.join();
            }
        }
    }
};