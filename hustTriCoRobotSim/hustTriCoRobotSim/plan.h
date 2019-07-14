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

// [��ͷģʽ] �켣��, ����λ�ú��ٶ���Ϣ
class HeadTrajectoryPoint {
public:
    Eigen::Affine2d pos;        // �켣��λ��
    double curvature;           // ���� (���ʰ뾶�ĵ���)
    double length;              // ����õ����·������
    Eigen::Vector2d vel_linear; // �������ٶ�
    double vel_rad;             // �������ٶ�
    double time;                // Ԥ�Ƶ���õ��ʱ��
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

    Eigen::Affine2d accurate_start;       // ��ȷ�����
    Eigen::Affine2d accurate_goal;        // ��ȷ���յ�
    std::array<int, 3> grid_start = { 0, 0, 0 };        // ��������
    std::array<int, 3> grid_goal = { 0, 0, 0 };         // ������յ�

    cv::Mat map;                // ������ͼ
    EnvironmentNAVXYTHETALAT *env = NULL;
    SBPLPlanner *planner = NULL;

    std::thread plan_thread;

    // ��һ�εĿ��й켣
    // ��Ŀ��㱻�ı�, �ᱻ����
    // ����ͼ�ı��, �ᱻ����
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> last_available_trajectory;

    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> temp_trajectory;
    Eigen::Vector2d temp_cls_point;

    std::mutex mutex_buffer;
    /* ���±�����Ҫ�������� */
    bool stop_plan = false;
    Eigen::Affine2d buffer_accurate_start;       // ��ȷ�����
    Eigen::Affine2d buffer_accurate_goal;        // ��ȷ���յ�
    std::array<int, 3> buffer_grid_start = { 0, 0, 0 };
    std::array<int, 3> buffer_grid_goal = {0, 0, 0};
    std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> solution_backend;
    std::condition_variable cv_has_solution;

    int last_plan_map_id = -1;       // ��һ�δ�GlobalMap��ȡ�������ڹ滮�ĵ�����ͼid
    int last_check_map_id = -1;      // ��һ�μ��û����ĵ�ͼid

    double safe_percent = 1;
    const double min_safe_percent_generate = 1.0;
    const double min_safe_percent_test = 0.90;

public:
    
    // ����㵽�����·����
    static bool nearest_path_point(
        const Eigen::Affine2d pos,
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        int* p_nearest_next = NULL,     // ��һ��·����
        double* p_err_linear = NULL,    // λ�ò�
        double* p_err_rad = NULL        // �ǶȲ�
    ) noexcept
    {
        if (path.empty()) {
            // ·��Ϊ��
            return false;
        }

        int nearest_next;
        double err_linear = std::numeric_limits<double>::infinity();
        double err_rad;

        Eigen::Vector2d car_xy = pos.translation();
        double car_angle = Eigen::Rotation2Dd(pos.rotation()).angle();

        if (path.size() == 1) {
            // �켣����Ϊ 1
            nearest_next = 0;
            err_linear = (car_xy - path[0].translation()).norm();
            err_rad = car_angle - Eigen::Rotation2Dd(path[0].rotation()).angle();
        }
        else {
            // �켣���� >= 2
            // ��ȡ�뵱ǰ����λ������Ĺ켣�߶�, ����¼λ��ƫ��
            for (int i = path.size() - 1; i > 0; i--) {
                // ǰ��·����
                Eigen::Affine2d next = path[i];
                Eigen::Affine2d prev = path[i - 1];
                // ·��������
                Eigen::Vector2d next_xy = next.translation();
                Eigen::Vector2d prev_xy = prev.translation();
                // ·���߶ε������ֱ�߾���
                double line_percent;
                double distance = min_dist_to_line_seg(car_xy, prev_xy, next_xy, NULL, &line_percent);
                if (distance < err_linear) {
                    // ��¼λ��ƫ��
                    nearest_next = i;
                    err_linear = distance;
                    // ����·���߶��ϵĽǶȲ�
                    double next_angle = Eigen::Rotation2Dd(path[i].rotation()).angle();
                    double prev_angle = Eigen::Rotation2Dd(path[i - 1].rotation()).angle();
                    double diff_angle = next_angle - prev_angle;
                    while (diff_angle > M_PI)
                        diff_angle -= 2 * M_PI;
                    while (diff_angle < -M_PI)
                        diff_angle += 2 * M_PI;
                    // ���㳵���Ƕ���켣�Ĳ�
                    err_rad = car_angle - (prev_angle + diff_angle * line_percent);
                }
            }
        }
        // �ǶȲ�, ����Ϊ[-pi, pi)
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

    // ��·��, �ϲ�����ת��ͬ�����ƽ��
    static std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> simplify_path(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        bool only_move = false,     // ������ת
        double tol_linear = 0.001, 
        double tol_rad = 0.1 / 180 * M_PI)
    {
        if (path.size() <= 2) {
            // ·�����ȹ���, �����
            return path;
        }

        std::vector<int> new_index;

        // �򻯺��·���������ԭʼ·����ͬ
        new_index.push_back(0);

        bool merge_rotate = false;   // ���ںϲ���ת
        bool merge_move = false;     // ���ںϲ�ƽ��
        for (size_t i = 1; i < path.size(); i++) {
            const Eigen::Affine2d& prev = path[new_index.back()];
            const Eigen::Affine2d& cur = path[i];

            double err_linear = (prev.translation() - cur.translation()).norm();
            double err_rad = std::abs(hust::diff_angle(prev, cur));

            if (!merge_rotate && !merge_move) {
                // ��δ�����ϲ�
                if (err_linear < tol_linear && err_rad < tol_rad)
                {
                    // ����ת��ƽ��
                    // �����õ�
                }
                else if (err_rad >= tol_rad && err_linear < tol_linear) {
                    // ����ת
                    if (!only_move) {
                        merge_rotate = true;
                    }
                } 
                else if (err_linear >= tol_linear && err_rad < tol_rad) {
                    // ��ƽ��
                    merge_move = true;
                }
                else {
                    // ��ת+ƽ��
                    if (!only_move) {
                        new_index.push_back(i);
                    }
                    else {
                        merge_move = true;
                    }
                }
            }
            else if (merge_rotate) {
                // ���ںϲ���ת
                if (err_linear >= tol_linear) {
                    // ��ƽ��, ����
                    i--;    
                    new_index.push_back(i);
                    merge_rotate = false;
                }
                else {
                    // ��ƽ��
                }
            }
            else if (merge_move) {
                // ���ںϲ�ƽ��
                if (err_rad >= tol_rad && !only_move) {
                    // ����ת, ����
                    i--;
                    new_index.push_back(i);
                    merge_move = false;
                }
                else {
                    // �жϵ��Ƿ���������
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
                        // ���е���������
                    }
                    else {
                        // �е���������, ����
                        i--;
                        new_index.push_back(i);
                        merge_move = false;
                    }
                }
            }
        }
        // �򻯺��·�����յ���ԭʼ·����ͬ
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

    // ��֤ĳ��������Ƿ�����
    bool is_pixel_free(const Eigen::Vector2d& p, uint8_t occupied_value = OccupyMap::PIXEL_NOT_PLAN) {
        // ת��Ϊ��ͼ����
        Eigen::Vector2i p_map = global_map.convert_to_map_pos(p);
        // ��֤���Ƿ��ڵ�ͼ��
        if (!global_map.check_position(p_map) ||
            map.at<uint8_t>(cv::Point(p_map[0], p_map[1])) >= 0xFF - occupied_value
        ) {
            // �ڲ����Ч��ռ�� Բ����Ч
            return false;
        }
        return true;
    }

    // ��֤ĳһ��ֱ���ϵ�����������Ƿ�����
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

    // ��֤ĳһ��ֱ���ϵ�����λ���Ƿ����
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

    // ��֤ĳ��λ���Ƿ���� , ��֤�����˵��ܳ��Ƿ�ռ��, ÿ���ֱ���ȡһ��
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

    // ��������Բ�ϴ��з���ĵ�, �����Ƿ�ΪԲ���Ƿ����
    // ���㷽��仯��Ӧ < 180��
    bool is_curve_passable(
        // �������
        const Eigen::Affine2d& arc_front, const Eigen::Affine2d& arc_back, 
        // �������
        double& arc_radius, Eigen::Vector2d& arc_center, double& arc_interplote_interval, 
        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& arc_interplote, 
        Eigen::Vector2d* bad_pos = NULL
    ) {
        // ��֤Բ��������յ��Ƿ���Ч
        if (!is_pos_passable(arc_front, bad_pos) || !is_pos_passable(arc_back, bad_pos)) {
            return false;
        }

        // �Ƕȱ仯�� +-
        double diff_angle = hust::sub_2pi(hust::get_angle(arc_front), hust::get_angle(arc_back));
        // �ҳ�
        double chord_length = (arc_back.translation() - arc_front.translation()).norm();
        // ���ʰ뾶 +-, ��ת�뾶Ϊ��, ��תΪ�� 
        double radius = (chord_length / 2) / std::sin(diff_angle / 2);
        // ����Բ��
        Eigen::Vector2d center = arc_front * Eigen::Vector2d(0, radius);
        // Բ����ʼ�Ƕ�
        Eigen::Vector2d c_to_p0 = arc_front.translation() - center;
        double begin_angle = std::atan2(c_to_p0[1], c_to_p0[0]);
        // Բ�������Ƕ�
        Eigen::Vector2d c_to_p1 = arc_back.translation() - center;
        double end_angle = std::atan2(c_to_p1[1], c_to_p1[0]);
        // ����
        double arc_length = std::abs(diff_angle * radius);
        // Բ���ڲ����
        int interplote_count = static_cast<int>(arc_length / global_map.resolution);
        // ��ʼ���ڲ������
        arc_interplote.clear();
        // �ڲ�ֵ����, ��֤�Ƿ��ڵ�ͼ������ײ
        bool is_collision = false;
        for (int interplote_i = 1; interplote_i <= interplote_count; interplote_i++) {
            double percent = 1.0 * interplote_i / (interplote_count + 1);
            double angle_of_circle = begin_angle + percent * (end_angle - begin_angle);     // �����Բ�ĵĽǶ�
            double angle_of_forward = hust::get_angle(arc_front) + percent * diff_angle;    // ��ͷ����
            // Ҫ��֤�ĵ��ȫ������
            Eigen::Vector2d p = center + Eigen::Vector2d(std::cos(angle_of_circle), std::sin(angle_of_circle)) * std::abs(radius);
            Eigen::Affine2d pos = Eigen::Translation2d(p) * Eigen::Rotation2Dd(angle_of_forward);
            if (is_pos_passable(pos, bad_pos)) {
                // �ڲ����Ч
                arc_interplote.push_back(pos);
            }
            else {
                // �ڲ���ڵ�ͼ���ռ�� Բ����Ч
                is_collision = true;
                break;
            }
        }
        if (!is_collision) {
            // ����ײ, Ϊ��Ч��Բ��
            arc_radius = radius;
            arc_center = center;
            arc_interplote_interval = arc_length / (interplote_count + 1);
            return true;
        }
        else {
            return false;
        }
    }

    // ���ĳһ���յ��ܷ�ʹ��Բ�����
    bool turn_has_passable_curve(
        // �������
        const Eigen::Vector2d& turn_point,
        double line_front_angle, double line_back_angle, 
        double max_shorten, double min_shorten, 
        // �������
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
            // Բ��ǰһ��·�����е�
            arc_front = Eigen::Translation2d(Eigen::Translation2d(turn_point) * Eigen::Rotation2Dd(line_front_angle) * Eigen::Vector2d(-shorten_dis, 0)) * Eigen::Rotation2Dd(line_front_angle);
            // Բ���һ��·�����е�
            arc_back = Eigen::Translation2d(Eigen::Translation2d(turn_point) * Eigen::Rotation2Dd(line_back_angle) * Eigen::Vector2d(shorten_dis, 0)) * Eigen::Rotation2Dd(line_back_angle);
            // ����Ƿ�Ϊ��Ч��Բ��
            if (is_curve_passable(arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote, cls_point)) {
                // ����ײ, Ϊ��Ч��Բ��
                return true;
            }
        }
        return false;
    }
    
    // [��ͷģʽ] ���ݼ򻯺��·��, ����Բ������, ���ɴ��������Լ�����Ĺ켣
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> generate_arc_trajectory(
        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> path
    ) {
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> opti_path;
        opti_path.emplace_back(path[0], 0, 0);

        if (path.size() >= 3) {
            // ����·������ >= 3ʱ����

            // ����ÿһ��·�����ۻ�����
            std::vector<double> length(path.size(), 0);
            for (size_t i = 1; i < path.size(); i++) {
                length[i] = length[i - 1] + (path[i - 1].translation() - path[i].translation()).norm();
            }

            // ����ÿһ���߶εĽǶ�
            for (size_t i = 0; i < path.size() - 1; i++) {
                Eigen::Vector2d diff_pos = path[i + 1].translation() - path[i].translation();
                double line_angle = std::atan2(diff_pos[1], diff_pos[0]);
                path[i] = Eigen::Translation2d(path[i].translation()) * Eigen::Rotation2Dd(line_angle);
            }

            //for (const Eigen::Affine2d& p : path) {
            //    opti_path.emplace_back(p, 0, 0);
            //}

            // �ϲ����ڵ����ɸ��յ�, �Ƕȱ仯Ӧ < 179��, �Ƕȱ仯����Ӧ��ͬ
            int merge_front = 1;
            while (merge_front < (int)path.size() - 1) {
                // ̰���㷨, һ���Ժϲ����ĵ�

                double front_diff_angle = hust::diff_angle(path[merge_front - 1], path[merge_front]);
                // �ڲ����ǵ�ͼ�������, ����һ�������ϲ����ٸ��յ�
                int max_merge_back = merge_front;
                for (int merge_back = merge_front + 1; merge_back < (int)path.size() - 1; merge_back++) {
                    // ���Ƕȱ仯�����Ƿ���ͬ
                    double diff_angle = hust::diff_angle(path[merge_back - 1], path[merge_back]);
                    if (diff_angle * front_diff_angle < 0) {
                        break;
                    }
                    // ���Ƕȱ仯�� < 179
                    double sum_diff_angle = hust::diff_angle(path[merge_front - 1], path[merge_back]);
                    if (std::abs(sum_diff_angle) > 179.0 / 180 * M_PI) {
                        break;
                    }
                    max_merge_back = merge_back;
                }

                // �Ӷൽ��, ���ǵ�ͼ�������, ���Ծ����ܶ�ĺϲ��յ�
                for (int merge_back = max_merge_back; merge_back >= merge_front; merge_back--) {
                    // ����ϲ���Ĺյ� �� ���ض̳���

                    // ��ʼ�յ�
                    Eigen::Affine2d p0 = Eigen::Translation2d(path[merge_front].translation()) * Eigen::Rotation2Dd(path[merge_front-1].rotation());
                    // �����յ�
                    Eigen::Affine2d p1 = Eigen::Translation2d(path[merge_back].translation()) * Eigen::Rotation2Dd(hust::get_angle(path[merge_back]) + M_PI);

                    // ʹ�����Ҷ�������¹յ㵽��ʼ�յ�ľ���
                    Eigen::Vector2d line = p1.translation() - p0.translation();
                    double line_angle = std::atan2(line[1], line[0]);
                    double beta = std::abs(hust::sub_2pi(line_angle, hust::get_angle(p1)));
                    double gamma = std::abs(hust::diff_angle(p0, p1));
                    double C = line.norm();
                    double B = std::sin(beta) / std::sin(gamma) * C;
                    double p0_to_turn_p = B;

                    // �¹յ��λ��
                    Eigen::Vector2d merged_turn_point = p0 * Eigen::Vector2d(p0_to_turn_p, 0);

                    // ����ǰ�����ض̳���
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

                    // ����ǰ����С�ض̳���
                    double front_min = (merged_turn_point - p0.translation()).norm();
                    double back_min = (merged_turn_point - p1.translation()).norm();
                    double min_shorten = std::max(front_min, back_min);

                    Eigen::Affine2d arc_front;          // Բ�����
                    Eigen::Affine2d arc_back;            // Բ���յ�
                    double arc_radius;                  // ���ʰ뾶
                    Eigen::Vector2d arc_center;         // Բ��Բ��
                    double arc_interplote_interval;     // Բ���ڲ���
                    std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> arc_interplote; // Բ���ڲ��

                    // ���ϲ���Ĺյ��Ƿ��ܹ�ʹ��Բ������
                    Eigen::Vector2d cls_point;
                    if (turn_has_passable_curve(
                        merged_turn_point,
                        hust::get_angle(path[merge_front - 1]), hust::get_angle(path[merge_back]),
                        max_shorten, min_shorten, 
                        arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote, &cls_point)
                    ) {
                        // �ҵ�����ЧԲ��

                        //opti_path.emplace_back(p0, 0, 0);
                        //opti_path.emplace_back(Eigen::Translation2d(merged_turn_point) * Eigen::Rotation2Dd(0), 0, 0);
                        //opti_path.emplace_back(p1, 0, 0);

                        // ����Բ��ͷ
                        opti_path.emplace_back(
                            arc_front,
                            1 / arc_radius,
                            opti_path.back().length + (opti_path.back().pos.translation() - arc_front.translation()).norm());
                        // ����Բ���ڲ��
                        for (const Eigen::Affine2d& p : arc_interplote) {
                            opti_path.emplace_back(
                                p,
                                1 / arc_radius,
                                opti_path.back().length + arc_interplote_interval);
                        }
                        // ����Բ���յ�
                        opti_path.emplace_back(
                            arc_back,
                            0,
                            opti_path.back().length + arc_interplote_interval);

                        merge_front = merge_back + 1;
                        break;
                    }
                    else {
                        if (merge_front == merge_back) {
                            // ʼ��δ�ҵ���ЧԲ��, ����
                            temp_cls_point = cls_point;
                            throw std::runtime_error("cannot find avalable arc!!!");
                        }
                    }
                }
            }
            


        //    // ����ÿһ���յ�����ض̳���
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

        //    // ���ݵ�ͼ���, ����ÿһ���յ��ʵ�����ض̳���
        //    for (size_t turn_i = 1; turn_i < path.size() - 1; turn_i++) {


        //        Eigen::Affine2d arc_front;          // Բ�����
        //        Eigen::Affine2d arc_back;            // Բ���յ�
        //        double arc_radius;                  // ���ʰ뾶
        //        Eigen::Vector2d arc_center;         // Բ��Բ��
        //        double arc_interplote_interval;     // Բ���ڲ���
        //        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> arc_interplote; // Բ���ڲ��

        //        // ���ùյ��Ƿ��ܹ�ʹ��Բ������
        //        if (turn_has_valid_curve(
        //            path[turn_i].translation(), 
        //            hust::get_angle(path[turn_i - 1]), hust::get_angle(path[turn_i]),
        //            max_shorten[turn_i], 
        //            arc_front, arc_back, arc_radius, arc_center, arc_interplote_interval, arc_interplote)
        //        ) {
        //            // �ҵ�����ЧԲ��
        //            // ����Բ��ͷ
        //            opti_path.emplace_back(
        //                arc_front,
        //                1 / arc_radius,
        //                opti_path.back().length + (opti_path.back().pos.translation() - arc_front.translation()).norm());
        //            // ����Բ���ڲ��
        //            for (const Eigen::Affine2d& p : arc_interplote) {
        //                opti_path.emplace_back(
        //                    p,
        //                    1 / arc_radius,
        //                    opti_path.back().length + arc_interplote_interval);
        //            }
        //            // ����Բ���յ�
        //            opti_path.emplace_back(
        //                arc_back,
        //                0,
        //                opti_path.back().length + arc_interplote_interval);
        //        }
        //        else {
        //            // δ�ҵ���ЧԲ��, ����
        //            throw std::runtime_error("cannot find avalable arc!!!");
        //        }
        //    }
        }

        // ����·���յ�
        opti_path.emplace_back(path.back(), 0,
            opti_path.back().length + (opti_path.back().pos.translation() - path.back().translation()).norm());

        return opti_path;
    }

    // ����㵽����Ĺ켣��
    static bool nearest_traj_point(
        const Eigen::Affine2d pos,
        const std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>>& path,
        int* p_nearest_next = NULL,     // ��һ��·����
        double* p_err_linear = NULL,    // λ�ò�
        double* p_err_rad = NULL        // �ǶȲ�
    ) noexcept
    {
        if (path.empty()) {
            // ·��Ϊ��
            return false;
        }

        int nearest_next;
        double err_linear = std::numeric_limits<double>::infinity();
        double err_rad;

        Eigen::Vector2d car_xy = pos.translation();
        double car_angle = Eigen::Rotation2Dd(pos.rotation()).angle();

        if (path.size() == 1) {
            // �켣����Ϊ 1
            nearest_next = 0;
            err_linear = (car_xy - path[0].pos.translation()).norm();
            err_rad = hust::diff_angle(pos, path[0].pos);
        }
        else {
            // �켣���� >= 2
            // ��ȡ�뵱ǰ����λ������Ĺ켣�߶�, ����¼λ��ƫ��
            for (int i = path.size() - 1; i > 0; i--) {
                // ǰ��·����
                Eigen::Affine2d next = path[i].pos;
                Eigen::Affine2d prev = path[i - 1].pos;
                // ·��������
                Eigen::Vector2d next_xy = next.translation();
                Eigen::Vector2d prev_xy = prev.translation();
                // ·���߶ε������ֱ�߾���
                double line_percent;
                double distance = min_dist_to_line_seg(car_xy, prev_xy, next_xy, NULL, &line_percent);
                if (distance < err_linear) {
                    // ��¼λ��ƫ��
                    nearest_next = i;
                    err_linear = distance;
                    // ����·���߶��ϵĽǶȲ�
                    double next_angle = Eigen::Rotation2Dd(path[i].pos.rotation()).angle();
                    double prev_angle = Eigen::Rotation2Dd(path[i - 1].pos.rotation()).angle();
                    double diff_angle = hust::sub_2pi(prev_angle, next_angle);
                    // ���㳵���Ƕ���켣�Ĳ�
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

    // [��ͷģʽ] �ڲ�켣, ��������Ϊ0��NaN���߶��Ͻ����ڲ�
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
                // ��Ҫ�ڲ�
                double p0_rad = hust::get_angle(p0.pos);
                double p1_rad = hust::get_angle(p1.pos);
                double diff_rad = hust::sub_2pi(p0_rad, p1_rad);
                for (int i = 0; i < interplote_count; i++) {
                    double percent = 1.0 * (i + 1) / (interplote_count + 1);
                    new_traj.emplace_back(
                        Eigen::Translation2d(p0.pos.translation() + line * percent) *
                        Eigen::Rotation2Dd(p0_rad + diff_rad * percent),
                        p0.curvature,                                   // ���ʸ������
                        p0.length + percent * line.norm()                           // ����
                    );
                }
            }
        }
        new_traj.push_back(origin_traj.back());
        return new_traj;
    }


    // [��ͷģʽ] ���ݶ���ѧԼ��, ʱ�������������ȷ�����������Ϣ�Ĺ켣
    std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> time_arg_trajectory(
        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> traj, 
        Eigen::Vector2d begin_vel_xy, // ��ǰƽ���ٶ�
        double begin_vel_rad,     // ��ǰ���ٶ�
        double max_accel_linear,// ����߼��ٶ�
        double max_accel_rad,   // ���Ǽ��ٶ�
        double max_vel_linear,
        double max_vel_rad,
        double tol_linear = 0.005,
        double tol_rad = 0.1 / 180 * M_PI
    ) {
        // ����������ļ��ٶ���·������, ��������ٶ�
        std::vector<double> max_speed(traj.size());
		std::vector<double> max_speed_rad(traj.size());
        // �������ٶ�Ϊ��ǰ�ٶ����˶������ϵ�ͶӰ, ��Ϊ��
        double begin_speed = begin_vel_xy.dot(traj[0].pos.rotation() * Eigen::Vector2d(1, 0));
        if (begin_speed < 0) {
            begin_speed = 0;
        }
        max_speed.front() = begin_speed;
		max_speed_rad.front() = begin_vel_rad;
        // ĩ���ٶ�Ϊ0
		max_speed.back() = 0;
		max_speed_rad.back() = 0;
		for (int i = 1; i < (int)traj.size() - 1; i++) {
            const auto& p = traj[i];
            if (p.curvature == 0 || std::isnan(p.curvature)) {
                // ����Ϊ0��δ֪, �ٶ�������
                max_speed[i] = std::numeric_limits<double>::infinity();
            }
            else if (std::isinf(p.curvature)) {
                // �������޴�, ͣ����ת
                max_speed[i] = 0;
            }
            else {
                // ������Ч
                double radius = std::abs(1 / p.curvature);  // ���ʰ뾶
                max_speed[i] = std::sqrt(max_accel_linear * radius);    // �������ļ��ٶ�sqrt(a*R)
            }
			max_speed_rad[i] = std::numeric_limits<double>::infinity();
        }

        // todo: ���ݽ��ٶȶ���ѧ����, ���������ٶ�

        // ��������ٶȲ���������ֵ
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

        // ���ݼ��ٶ�����, ��ǰ����, ����ÿ���켣����ٶ�����
        for (int i = traj.size() - 2; i >= 0; i--) {
            // ƽ�ƾ���
            double distance = traj[i + 1].length - traj[i].length;
            // ��һ������ٶ�
            double vel_next = max_speed[i + 1];
            // ���ٶ���Ϊ�����ٶ�
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_prev = std::sqrt(2 * max_accel_linear * distance + vel_next * vel_next);
            if (max_speed[i] > vel_prev) {
                max_speed[i] = vel_prev;
            }
        }
        // ���ݼ��ٶ�����, ��󴫲�
        for (size_t i = 1; i < traj.size(); i++) {
            // ƽ�ƾ���
            double distance = traj[i].length - traj[i - 1].length;
            // ��һ������ٶ�
            double vel_prev = max_speed[i - 1];
            // ���ٶ���Ϊ�����ٶ�
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_next = std::sqrt(2 * max_accel_linear * distance + vel_prev * vel_prev);
            if (max_speed[i] > vel_next) {
                max_speed[i] = vel_next;
            }
			// ����Ƕȱ仯��
			double diff_angle = hust::diff_angle(traj[i].pos, traj[i - 1].pos);
			// ��һ��Ľ��ٶ�
			double rad_prev = max_speed_rad[i - 1];
			// �������ٶȼ�������ʱ��
			double rad_next = std::sqrt(2 * max_accel_rad * std::abs(diff_angle) + rad_prev * rad_prev);
			if (max_speed_rad[i] > rad_next) {
				max_speed_rad[i] = rad_next;
			}
        }

        // ���������ٶ��Լ�����, ����ÿһ���Ԥ��ʱ��
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

			// ȡ�������ɶ��н�����ʱ��
			if (dt_linear > dt_rad) {
				traj[i].time = traj[i - 1].time + dt_rad;
			}
			else {
				traj[i].time = traj[i - 1].time + dt_linear;
			}
        }

        // ����ʱ����ǶȲ�, �����ٶ�����ٶ�ǰ��
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

    // [������] �ڲ巽ʽ
    enum class PathInterplateType {
        SAME_DISTANCE,      // �Ⱦ�
        ALIGNED_TO_GRID,    // ���뵽X��Y����
    };

    // [������] �ڹ켣�Ͻ����ڲ�
    static std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> interplate_path(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path,
        double resolution,      // �ڲ���
        PathInterplateType type = PathInterplateType::SAME_DISTANCE // �ڲ�����
    ) {
        if (path.size() < 2) {
            return path;
        }

        std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> new_path;
        if (type == PathInterplateType::SAME_DISTANCE) {
            // �Ⱦ��ڲ�
            for (size_t i = 0; i < path.size() - 1; i++) {
                new_path.push_back(path[i]);
                double distance = (path[i].translation() - path[i + 1].translation()).norm();
                int count = static_cast<int>(std::ceil(distance / resolution));
                if (count > 1) {
                    // ��Ҫ�ڲ�
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
            // ���뵽X��Y�����ڲ�
            // todo
        }
        return new_path;
    }

    // [������] �����ڲ���·�����ɴ����ٶȵĹ켣
    static std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> generate_trajectory(
        const std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>>& path, 
        Eigen::Vector2d cur_vel_xy, // ��ǰƽ���ٶ�
        double cur_vel_rad,     // ��ǰ���ٶ�
        double max_accel_linear,// ����߼��ٶ�
        double max_accel_rad,   // ���Ǽ��ٶ�
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

        // ��·������
        double total_length = 0;
        traj[0].length = 0;
        for (size_t i = 1; i < traj.size(); i++) {
            total_length += (traj[i - 1].pos.translation() - traj[i].pos.translation()).norm();
            traj[i].length = total_length;
        }

        // �й���������Сת��뾶 0.2 �������������ٶȷ���
        double turn_curvature = 1 / 0.2;
        traj[0].curvature = 0;
        // �ٶ�ΪͶӰ
        traj[0].vel_linear = (traj[1].pos.translation() - traj[0].pos.translation()).normalized() * ((traj[1].pos.translation() - traj[0].pos.translation()).normalized().dot(cur_vel_xy));
        for (size_t i = 1; i < traj.size() - 1; i++) {
            // ǰ��������
            Eigen::Vector2d p0 = traj[i - 1].pos.translation();
            Eigen::Vector2d p1 = traj[i].pos.translation();
            Eigen::Vector2d p2 = traj[i + 1].pos.translation();
            // �����߶�
            Eigen::Vector2d l_01 = p1 - p0;
            Eigen::Vector2d l_12 = p2 - p1;
            // ����߶γ���
            if (l_01.norm() < 0.005 || l_12.norm() < 0.005) {
                // �߶ι���
                traj[i].curvature = turn_curvature;
            }
            else {
                // �����߶εĽǶ�
                double angle_line_01 = std::atan2(l_01[1], l_01[0]);
                double angle_line_12 = std::atan2(l_12[1], l_12[0]);
                // �����߶εļн�
                double diff_angle = hust::sub_2pi(angle_line_01, angle_line_12);
                if (std::abs(diff_angle) > 50 * M_PI / 180) {
                    // ��ת��, ��Ҫͣ��
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
                    // �й���
                    traj[i].curvature = turn_curvature;
                    traj[i].vel_linear = l_12.normalized() * 1.5;
                }
                else {
                    // ֱ��
                    traj[i].curvature = 0;
                    traj[i].vel_linear = l_12.normalized() * max_vel_linear;
                }
            }
        }
        // ���һ���ٶ�Ϊ0
        traj.back().curvature = 0;
        traj.back().vel_linear = Eigen::Vector2d(0, 0);

        // ���ݼ��ٶ�����, ��ǰ����, ����ÿ���켣����ٶ�����
        for (int i = traj.size() - 2; i >= 0; i--) {
            // ƽ�ƾ���
            double distance = traj[i + 1].length - traj[i].length;
            // ��һ������ٶ�
            double vel_next = traj[i+1].vel_linear.norm();
            // ���ٶ���Ϊ�����ٶ�
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_prev = std::sqrt(2 * max_accel_linear * distance + vel_next * vel_next);
            if (traj[i].vel_linear.norm() > vel_prev) {
                traj[i].vel_linear = traj[i].vel_linear.normalized() * vel_prev;
            }
        }
        // ���ݼ��ٶ�����, ��󴫲�
        for (size_t i = 1; i < traj.size(); i++) {
            // ƽ�ƾ���
            double distance = traj[i].length - traj[i-1].length;
            // ��һ������ٶ�
            double vel_prev = traj[i-1].vel_linear.norm();
            // ���ٶ���Ϊ�����ٶ�
            // distance = (v_a + v_b) / 2 * t
            // v_b = v_a + a * t
            // => v_b = sqrt(2 * a * distance + v_a^2)
            double vel_next = std::sqrt(2 * max_accel_linear * distance + vel_prev * vel_prev);
            if (traj[i].vel_linear.norm() > vel_next) {
                traj[i].vel_linear = traj[i].vel_linear.normalized() * vel_next;
            }
        }

        // ���������ٶ��Լ�����, ����ÿһ���Ԥ��ʱ��
        traj[0].time = 0;
        for (size_t i = 1; i < traj.size(); i++) {
            double avg_vel = (traj[i].vel_linear.norm() + traj[i - 1].vel_linear.norm()) / 2;
            double distance = traj[i].length - traj[i - 1].length;
            traj[i].time = traj[i - 1].time + distance / avg_vel;
        }

        // ����ʱ����ǶȲ�, ������ٶ�ǰ��
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

    // ����㵽�߶ε���С����
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

    // ��ȡĬ�ϵĳ���������
    static std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> get_default_car() {
        static std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> default_car;
        // ����Ĭ�ϳ�ʼ��Ϊ 0.6m �� 0.8m ��
        double r_x = 0.4;
        double r_y = 0.3;
        // ��ǰ
        default_car.emplace_back(r_x, r_y);
        // ���
        default_car.emplace_back(r_x, -r_y);
        // �Һ�
        default_car.emplace_back(-r_x, -r_y);
        // ��ǰ
        default_car.emplace_back(-r_x, r_y);
        return default_car;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // �����ȱ�ʾ�ĽǶ�, ת�����Ƕȱ��
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

    // ���Ƕȱ��ת�������ȱ�ʾ�ĽǶ�
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

    // ����������ת������ɢ״̬�ռ�����
    std::array<int, 3> convert_world_to_statepos(const Eigen::Affine2d &world_pos) {
        std::array<int, 3> state_pos;
        // ����
        Eigen::Vector2i xy = global_map.convert_to_map_pos(world_pos.translation());
        state_pos[0] = xy[0];
        state_pos[1] = xy[1];
        // ��ת
        double theta = Eigen::Rotation2Dd(world_pos.rotation()).angle();
        state_pos[2] = convert_angle_to_id(theta);
        return state_pos;
    }

    // ����ɢ״̬�ռ�����ת������������
    Eigen::Affine2d convert_statepos_to_world(const std::array<int, 3> &state_pos) {
        Eigen::Vector2d xy = global_map.convert_to_world_pos(Eigen::Vector2i(state_pos[0], state_pos[1]));
        return Eigen::Translation2d(xy) * Eigen::Rotation2Dd(convert_id_to_angle(state_pos[2]));
    }

    Planner(
        GlobalMap &_global_map,
        const std::string& motion_prim_path,
        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& car = get_default_car(), 
        double vel_linear = 2,  // �滮�����ڼ���ƽ�ƴ���
        double vel_rad = 2 * M_PI   // �滮�����ڼ�����ת����
    ) : 
        global_map(_global_map), 
        car(car)
    {

        // ���㳵�����뾶
        double addition_ratio = 1.0;
        for (const Eigen::Vector2d& p : car) {
            car_perimeter.emplace_back(p[0] * addition_ratio, p[1] * addition_ratio);

            double radius = p.norm() * addition_ratio;
            if (car_max_radius == 0 || radius > car_max_radius) {
                car_max_radius = radius;
            }
        }
        // ���㳵����С�뾶
        for (size_t i = 0; i < car.size(); i++) {
            double radius = min_dist_to_line_seg({ 0, 0 }, car[i], car[(i + 1) % car.size()]);
            if (car_min_radius == 0 || radius < car_min_radius) {
                car_min_radius = radius;
            }
        }

        // Ϊ�˸�˳��, ���뾶��СһЩ
        //car_max_radius = car_min_radius + 0.06;

        // ��ȡ������ͼ
        global_map.set_plan_map_params(car_min_radius, car_max_radius);
        auto id_map = global_map.generate_plan_map();
        last_plan_map_id = id_map.first;
        map = 0xFF - id_map.second;
        map.setTo(0xFF - OccupyMap::PIXEL_FREE, map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));

        // ��ʼ���滮����
        env = new EnvironmentNAVXYTHETALAT();

        // ����ֵ >= 0xFF - PIXEL_NOT_PLAN ������Ϊ���ɹ滮����, ��Ϊ�ڸ������ڻ����˿϶��������ϰ���
        if (!env->SetEnvParameter("cost_inscribed_thresh", 0xFF - OccupyMap::PIXEL_NOT_PLAN)) {
            throw SBPL_Exception("ERROR: failed to set parameters");
        }
        // ����ֵ >= 0xFF - PIXEL_MAY_CLS �������ڻ����˿��ܻ������ϰ���
        if (!env->SetEnvParameter("cost_possibly_circumscribed_thresh", 0xFF - OccupyMap::PIXEL_MAY_CLS)) {
            throw SBPL_Exception("ERROR: failed to set parameters");
        }

        // ��ͼ������ֵ >= 0xFF - PIXEL_LIMITED, ��ʾΪ�ϰ���
        if (!env->InitializeEnv(
            map.cols, map.rows, map.data,
            // ���
            grid_start[0] * global_map.resolution,
            grid_start[1] * global_map.resolution,
            grid_start[2] * (2 * M_PI / angle_number), 
            // �յ�
            grid_goal[0] * global_map.resolution,
            grid_goal[1] * global_map.resolution,
            grid_goal[2] * (2 * M_PI / angle_number), 
            // �յ��ݲ�
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

        // ����һ���滮�� ARA�滮��, ��������
        planner = new ARAPlanner(env, true);
        // �趨�滮���Ŀ�ʼ�ͽ���״̬
        if (planner->set_start(MDPCfg.startstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set start state");
        }
        if (planner->set_goal(MDPCfg.goalstateid) == 0) {
            throw SBPL_Exception("ERROR: failed to set goal state");
        }
        // �趨��ʼ�������޶�Ϊ���Ž��n��
        planner->set_initialsolution_eps(1.2);
        // �����ҵ���ʼ��ŷ���, ��ʱ�ͷ���
        planner->set_search_mode(false);
    }

    ~Planner() {
        if (plan_thread.joinable()) {
            plan_thread.join();
        }

        delete planner;
        delete env;
    }

    // ֪ͨ�滮�� ��㷢���˱仯
    bool set_start(const Eigen::Affine2d& start, std::array<int, 3> &new_start = std::array<int, 3>{ {0, 0, 0}}) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        // ת������ɢ״̬�ռ�
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
    // ֪ͨ�滮�� Ŀ�귢���˱仯
    bool set_goal(const Eigen::Affine2d& goal, std::array<int, 3> &new_goal = std::array<int, 3>{ { 0, 0, 0 } }) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        // ת������ɢ״̬�ռ�
        new_goal = convert_world_to_statepos(goal);
        if (buffer_grid_goal != new_goal) {
            buffer_grid_goal = new_goal;
            buffer_accurate_goal = goal;
            // ���յ㷢���˸���, ��մ�ǰ����Ч�켣����
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

    // ���ݸ������, ��ǰ�ٶ��� �յ�, ���Ի�ȡ�滮�ĵ�ͼ
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

        // ���¹滮���
        set_start(car_pos);

        // ���¹滮�յ�
        set_goal(target_pos);

        std::vector<HeadTrajectoryPoint, Eigen::aligned_allocator<HeadTrajectoryPoint>> trajectory;

        err_linear = (car_pos.translation() - target_pos.translation()).norm();
        err_rad = hust::diff_angle(car_pos, target_pos);
        if (err_linear < 0.05) {
            // ���ӽ��յ�, ֱ�ӹ�ȥ
            trajectory.push_back(HeadTrajectoryPoint(car_pos));
            trajectory.push_back(HeadTrajectoryPoint(target_pos));
            return trajectory;
        }

        mutex_buffer.lock();
        if (!last_available_trajectory.empty() && global_map.is_newest_plan_map(last_check_map_id)) {
            // ������Ч�켣����, ֱ��ʹ�û���켣
            trajectory = last_available_trajectory;
            mutex_buffer.unlock();
        }
        else {
            mutex_buffer.unlock();

            // û����Ч�켣����, ��Ҫ�������ɹ켣
            if (!global_map.is_newest_plan_map(last_check_map_id)) {
                // ��ǰ������ͼ����
                // ���µ�ͼ
                global_map.set_plan_map_params(car_min_radius, car_max_radius);
                auto id_map = global_map.generate_plan_map();
                last_check_map_id = id_map.first;

                cv::Mat new_map(0xFF - id_map.second);
                new_map.setTo(0xFF - OccupyMap::PIXEL_FREE, new_map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));
                new_map.copyTo(map);
                hust::log("[PLAN] Re-generate plan map to check trajectory, id=" + std::to_string(id_map.first));
            }

            // ��ȡ���¹滮��·��
            mutex_buffer.lock();
            auto path = solution_backend;
            mutex_buffer.unlock();

            if (path.empty()) {
                // �޽�
                hust::log_status("[PLAN] Empty solution path");
                trajectory = {};
                last_available_trajectory = {};
                return trajectory;
            }

            if (target_pos.translation()[0] < -2) {
                hust::log_status("[PLAN] task3 target");
            }

            // ���·����Ŀ��λ�õĲ�
            if (false) {
                if (nearest_path_point(target_pos, path, &nearest_traj_i, &err_linear, &err_rad)) {
                    if (nearest_traj_i < 1) {
                        // ��֪·��, �޷�����Ŀ���, �޽�
                        hust::log_status("[PLAN] path cannot reach goal (nearest_traj_i < 1)");
                        trajectory = {};
                        last_available_trajectory = {};
                        return trajectory;
                    }
                    else if (err_linear > 0.1 || std::abs(err_rad) > M_PI / 4) {
                        // ���켣λ�õ�Ŀ��λ�÷�ɴ�
                        // todo: �����ϽǶȱ仯
                        if (!is_line_free(target_pos.translation(), path[nearest_traj_i - 1].translation())) {
                            // ���ɴ�, �޽�
                            hust::log_status("[PLAN] path cannot reach goal (occupied)");
                            trajectory = {};
                            last_available_trajectory = {};
                            return trajectory;
                        }
                    }
                }
                else {
                    // ��Ӧ�ó��ָ������
                    throw std::runtime_error("cannot find nearest_traj_point to target_pos");
                }

                // ɾ�������·����
                path.erase(path.begin() + nearest_traj_i, path.end());
                // ��Ŀ�����Ϊ·���յ�
                path.push_back(target_pos);
            }
            else {
                err_linear = (target_pos.translation() - path.back().translation()).norm();
                err_rad = hust::diff_angle(target_pos, path.back()); 
                if (err_linear > 0.1 || std::abs(err_rad) > M_PI / 4) {
                    // ���켣λ�õ�Ŀ��λ�÷�ɴ�
                    // todo: �����ϽǶȱ仯
                    if (!is_line_free(target_pos.translation(), path.back().translation())) {
                        // ���ɴ�, �޽�
                        hust::log_status("[PLAN] path cannot reach goal (occupied)");
                        trajectory = {};
                        last_available_trajectory = {};
                        return trajectory;
                    }
                }
            }

            for (int n = 10; n >= 0; n--) {
                safe_percent = min_safe_percent_generate + n / 10.0 * (1 - min_safe_percent_generate);

                // ���ɹ켣, ������·���Ƿ�, �޷�������Ч�켣, ����׳��쳣
                try {
                    trajectory = generate_arc_trajectory(path); // ��ʱ!!!
                    temp_trajectory = trajectory;
                    break;
                }
                catch (const std::exception& err) {
                    if (n == 0) {
                        // ·����ռ��, �޽�
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

                // ���켣�Ƿ����
                bool is_occupied = false;
                for (size_t i = 0; i < trajectory.size() - 1; i++) {
                    Eigen::Vector2d cls_point;
                    if (!is_line_passable(trajectory[i].pos, trajectory[i + 1].pos, false, &cls_point)) {
                        if (n == 0) {
                            // ·����ռ��, �޽�
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

            // ������Ч�켣����
            hust::log("[PLAN] Last trajectory is ok id=" + std::to_string(last_check_map_id));
            mutex_buffer.lock();
            last_available_trajectory = trajectory;
            mutex_buffer.unlock();
        }

        // �ڹ켣�е�ֱ���߶��Ͻ����ڲ�
        trajectory = interplote_trajectory(trajectory);

        // ��þ��뵱ǰλ������Ĺ켣��
        if (nearest_traj_point(car_pos, trajectory, &nearest_traj_i, &err_linear, &err_rad)) {
            // ɾ������·��
            double passed_length = trajectory[nearest_traj_i].length;
            trajectory.erase(trajectory.begin(), trajectory.begin() + nearest_traj_i);
            for (HeadTrajectoryPoint& p : trajectory) {
                p.length -= passed_length;
            }
        }
        else {
            // ��Ӧ�ó��ָ������
            throw std::runtime_error("cannot find nearest_traj_point to car_pos");
        }

        // ��鵱ǰλ�õ��켣����Ƿ�ɴ�
        // todo: �����ϽǶȱ仯
        if (!is_line_passable(car_pos, trajectory.front().pos) && is_pos_passable(car_pos)) {
            // ���ɴ�, �޽�
            hust::log_status("[PLAN] car cannot reach traj start");
            trajectory = {};
            last_available_trajectory = {};
            return trajectory;
        }

        // ���뵱ǰλ����Ϊ��ʵ���
        double cur_to_front = (car_pos.translation() - trajectory.front().pos.translation()).norm();
        for (HeadTrajectoryPoint& p : trajectory) {
            p.length += cur_to_front;
        }
        trajectory.insert(
            trajectory.begin(),
            HeadTrajectoryPoint(car_pos, std::numeric_limits<double>::signaling_NaN(), 0)
        );

        // ��ԭ�յ�λ��
        trajectory.back().pos = Eigen::Translation2d(trajectory.back().pos.translation()) * Eigen::Rotation2Dd(target_pos.rotation());

        // ʱ��������켣
        trajectory = time_arg_trajectory(trajectory,
            Eigen::Vector2d(car_speed[0], car_speed[1]), car_speed[2],
            max_accel_linear, max_accel_rad,
            max_vel_linear, max_vel_rad);

        hust::log_status("[PLAN] OK");
        temp_trajectory = trajectory;
        return trajectory;
    }

    // �ڲ��滮�߳�
    void _thread_plan() {
        std::vector<int> solution_state;
        while (true) {

            mutex_buffer.lock();
            // ����Ƿ��˳�
            if (stop_plan) {
                stop_plan = true;
                mutex_buffer.unlock();
                break;
            }
            // ��ʼλ���и���
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
            // Ŀ��λ���и���
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
                // ��ʼ��Ŀ��λ�÷��������, ���¹滮��������
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

            // ���滮��ͼ�Ƿ��и���
            global_map.set_plan_map_params(car_min_radius, car_max_radius);
            auto id_map = global_map.generate_plan_map();
            int id = id_map.first;
            mutex_buffer.lock();
            if (last_plan_map_id != id) {
                // ������ͼ�б仯
                last_plan_map_id = id;
                mutex_buffer.unlock();

                cv::Mat new_map(0xFF - id_map.second);
                new_map.setTo(0xFF - OccupyMap::PIXEL_FREE, new_map <= (0xFF - OccupyMap::PIXEL_UNKNOWN));
                new_map.copyTo(map);
                // ARA�滮����֧�������ĵ�ͼ����
                env->SetMap(map.data);
                ((ARAPlanner*)planner)->costs_changed();
                hust::log("[PLAN] Fully replan because map changed!!! id=" + std::to_string(last_plan_map_id));
            }
            else {
                mutex_buffer.unlock();
            }

            // ���й滮
            int ret = planner->replan(0.1, &solution_state);
            if (ret == 1) {
                // �ɹ��ҵ���
                // ��״̬�ռ�Ľ�ת����X Y Theta�ռ�
                std::vector<sbpl_xy_theta_pt_t> path;
                env->ConvertStateIDPathintoXYThetaPath(&solution_state, &path);

                std::vector<Eigen::Affine2d, Eigen::aligned_allocator<Eigen::Affine2d>> solution;

                // ����滮���
                for (const sbpl_xy_theta_pt_t &p : path) {
                    // ת������������ϵ Eigen::Affine2d
                    solution.emplace_back(
                        Eigen::Translation2d(p.x + global_map.left_bottom[0], p.y + global_map.left_bottom[1]) * 
                        Eigen::Rotation2Dd(p.theta));
                }

                // �򻯹滮���
                solution = simplify_path(solution, true);

                mutex_buffer.lock();
                solution_backend = solution;
                mutex_buffer.unlock();
                cv_has_solution.notify_all();
            }
            else {
                // �޽�
                //mutex_buffer.lock();
                //solution_backend.clear();
                //mutex_buffer.unlock();
            }
        }
    }

    // ��ʼ�����߳��н��й滮
    void start_plannig() {
        if (!plan_thread.joinable()) {
            plan_thread = std::thread(std::bind(&Planner::_thread_plan, this));
        }
    }

    // �ȴ���ȡ��
    void wait_for_solution() {
        std::unique_lock<std::mutex> lk(mutex_buffer);
        cv_has_solution.wait(lk, [this] {return solution_backend.size() > 0; });
    }

    // �����滮�߳�
    void stop_planning(bool wait_for_stop = false) {
        std::lock_guard<std::mutex> lock(mutex_buffer);
        stop_plan = true;
        // �ȴ��߳̽���
        if (wait_for_stop) {
            if (plan_thread.joinable()) {
                plan_thread.join();
            }
        }
    }
};