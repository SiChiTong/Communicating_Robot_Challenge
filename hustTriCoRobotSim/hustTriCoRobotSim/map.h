#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <vector>
#include <set>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <mutex>
#include <algorithm>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "main.h"

static Eigen::Vector3d GetRPYFromRotateMatrix(const Eigen::Matrix3d &matrix) {
    Eigen::Vector3d rpy = matrix.eulerAngles(2, 1, 0);
    std::swap(rpy[0], rpy[2]);
    // current euler angle range: [-pi, pi] x [-pi, pi] x [0, pi]
    // desired euler angle range: [-pi, pi] x [-pi / 2, pi / 2] x [-pi, pi]
    if (rpy[1] < -M_PI / 2) {
        rpy[1] = -M_PI - rpy[1];
        rpy[2] -= M_PI;
        rpy[0] -= M_PI;
        if (rpy[0] < -M_PI) {
            rpy[0] += 2 * M_PI;
        }
    }
    else if (rpy[1] > M_PI / 2) {
        rpy[1] = M_PI - rpy[1];
        rpy[2] -= M_PI;
        rpy[0] -= M_PI;
        if (rpy[0] < -M_PI) {
            rpy[0] += 2 * M_PI;
        }
    }
    return rpy;
}

// 基础地图类
class MapBase
{
public:
    const double resolution;
    const Eigen::Vector2i size;
    const Eigen::Vector2d left_bottom;
    const Eigen::Vector2d right_top;
    cv::Mat cv_data;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    MapBase(const Eigen::Vector2d &corner_left_bottom, 
        const Eigen::Vector2d &corner_right_top, 
        double resolution, uint8_t value = 0x00) 
        :
        size(
            static_cast<int>(std::round((corner_right_top[0] - corner_left_bottom[0]) / resolution)),
            static_cast<int>(std::round((corner_right_top[1] - corner_left_bottom[1]) / resolution))),
        resolution(resolution),
        left_bottom(corner_left_bottom), 
        right_top(corner_left_bottom + size.cast<double>() * resolution),
        cv_data(size[1], size[0], CV_8UC1, value)
    {
        if (size[0] % 4) {
            std::cout << "map width must be divided by 4" << std::endl;
            throw std::invalid_argument("map width must be divided by 4");
        }
    }
    MapBase(const MapBase&) = delete; // 禁止 拷贝构造
    MapBase& operator=(const MapBase&) = delete; // 禁止 拷贝赋值

    MapBase(MapBase&&) = default;   // 移动构造
    MapBase& operator=(MapBase&& other) = default;  // 移动赋值


    /* 数据读取 */
    const uint8_t *get_data() const {
        return this->cv_data.data;
    }

    // 带有越界检查的地图数据读取
    uint8_t get(const Eigen::Vector2i &pos) const {
        if (!this->check_position(pos)) {
            std::stringstream ss;
            ss << "Map range: ([0, " << size[0] << "], [0, " << size[1] << "]), given index: (" << pos[0] << ", " << pos[1] << ")";
            throw std::out_of_range(ss.str());
        }
        return this->_get(pos);
    }

    // float -0.5 0.5 1.5 2.5
    //       --|---|---|---|-...
    // int       0   1   2

    // 带有越界检查的地图数据写入
    void set(const Eigen::Vector2i &pos, uint8_t value) {
        if (!this->check_position(pos)) {
            std::stringstream ss;
            ss << "Map range: ([0, " << size[0] << "], [0, " << size[1] << "]), given index: (" << pos[0] << ", " << pos[1] << ")";
            throw std::out_of_range(ss.str());
        }
        this->_set(pos, value);
    }

    // 由double类型世界坐标 获取对应的int类型地图坐标
    Eigen::Vector2i convert_to_map_pos(const Eigen::Vector2d &pos_in_world) const {
        Eigen::Vector2d pos = (pos_in_world - left_bottom) / resolution;
        return Eigen::Vector2i(static_cast<int>(std::round(pos[0])), static_cast<int>(std::round(pos[1])));
    }

    // 由int类型地图坐标 转换为 double类型世界坐标
    Eigen::Vector2d convert_to_world_pos(const Eigen::Vector2i &map_pos) const {
        return map_pos.cast<double>() * resolution + left_bottom;
    }

    // 检查越界
    bool check_position(const Eigen::Vector2i &pos) const {
        if (pos[0] < 0 || pos[0] >= size[0]) {
            return false;
        }
        else if (pos[1] < 0 || pos[1] >= size[1]) {
            return false;
        }
        else {
            return true;
        }
    }

    // 保存地图 需要写全路径 文件名 文件后缀
    // 由于坐标原因, 保存和读取时需要上下翻转
    bool save(const std::string &file_path, bool flip_y=true) const {
        if (flip_y) {
            cv::Mat fliped;
            cv::flip(cv_data, fliped, 0);
            return cv::imwrite(file_path, fliped);
        }
        else {
            return cv::imwrite(file_path, cv_data);
        }
    }

    // 读取地图
    void load(const std::string& file_path, bool flip_y = true) {
        cv::Mat origin_map = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
        if (origin_map.size() != cv_data.size()) {
            throw std::invalid_argument("origin_map size must equal to cv_data");
        }
        if (flip_y) {
            cv::flip(origin_map, origin_map, 0);
        }
        origin_map.copyTo(cv_data);
    }


    /* 以下函数直接读写数据, 不检查越界 */
    uint8_t _get(const Eigen::Vector2i &pos) const {
        return this->_get(pos[0], pos[1]);
    }
    uint8_t _get(int x, int y) const {
        return this->cv_data.data[x + y * size[0]];
    }
    void _set(const Eigen::Vector2i &pos, uint8_t value) {
        this->_set(pos[0], pos[1], value);
    }
    void _set(int x, int y, uint8_t value) {
        this->cv_data.data[x + y * size[0]] = value;
    }
};

// 占据栅格地图
class OccupyMap : public MapBase {

public:
    static const uint8_t PIXEL_OCCUPIED =   0x00;   // 被实体占据
    static const uint8_t PIXEL_LIMITED =    0x50;   // 由于障碍物空间外形等原因, 而不可进入的区域
    static const uint8_t PIXEL_NOT_PLAN =   0x80;   // 由于车身尺寸和安全距离产生的绝对不可导航区域
    static const uint8_t PIXEL_MAY_CLS =    0xA0;   // 在该区域内行驶, 由于车身尺寸, 可能发生碰撞
    static const uint8_t PIXEL_UNKNOWN =    0xE0;   // 未知
    static const uint8_t PIXEL_FREE =       0xFF;   // 自由

    // 使用基类构造函数
    using MapBase::MapBase;

    // 三维旋转投影到二维
    static Eigen::Affine2d pos3d_to_2d(const Eigen::Affine3d pos3d) {
        Eigen::Vector3d rpy = GetRPYFromRotateMatrix(pos3d.rotation());
        Eigen::Affine2d pos2d = Eigen::Affine2d(
            Eigen::Translation2d(
                pos3d.translation()[0],
                pos3d.translation()[1]))
            .rotate(
                Eigen::Rotation2Dd(rpy[2])
            );
        return pos2d;
    }

    static Eigen::Affine3d pos2d_to_3d(const Eigen::Affine2d pos2d) {
        Eigen::Affine3d pos3d = Eigen::Affine3d(
            Eigen::Translation3d(
                pos2d.translation()[0], 
                pos2d.translation()[1]))
            .rotate(
                Eigen::AngleAxisd(
                    Eigen::Rotation2Dd(pos2d.rotation()).angle(), 
                    Eigen::Vector3d(0, 0, 1)
                )
            );
        return pos3d;
    }
};

// 家具的子地图
class FurnitureMap : public OccupyMap
{
public:
    static const cv::Size2d chair_size;
    static const cv::Size2d desk_size;
    static const cv::Size2d cabilnet_size;
    // 任务二家具原始位置范围
    static const Eigen::Vector2d task2_src_left_bottom;
    static const Eigen::Vector2d task2_src_right_top;
    // 任务二家具目标位置范围
    static const Eigen::Vector2d task2_dst_left_bottom;
    static const Eigen::Vector2d task2_dst_right_top;
    // 任务二家具随机生成位置范围
    static const Eigen::Vector2d task2_random_left_bottom;
    static const Eigen::Vector2d task2_random_right_top;
private:
    // 记录预先读取的家具地图
    static cv::Mat cv_chair;
    static cv::Mat cv_desk;
    static cv::Mat cv_cabilnet;
    static Eigen::Vector2d load_size;
    static double load_resolution;
public:
    // 读取预先记录的家具地图
    // 灰度值 0: 实体占据, 255: 自由, 其他值[1, 254]: 不可进入
    static void load_furnitures_map(std::string file_path, double resolution = 0.02) {
        if (cv_desk.empty() || cv_chair.empty() || cv_cabilnet.empty()) {
            cv::Mat chair = cv::imread(file_path + "chair.png", cv::IMREAD_GRAYSCALE);
            cv::Mat desk = cv::imread(file_path + "desk.png", cv::IMREAD_GRAYSCALE);
            cv::Mat cabilnet = cv::imread(file_path + "cabilnet.png", cv::IMREAD_GRAYSCALE);

            if (chair.type() != CV_8UC1) {
                throw std::invalid_argument("furnitures_map cv_chair must be CV_8UC1");
            }
            if (desk.type() != CV_8UC1) {
                throw std::invalid_argument("furnitures_map cv_desk must be CV_8UC1");
            }
            if (cabilnet.type() != CV_8UC1) {
                throw std::invalid_argument("furnitures_map cv_cabilnet must be CV_8UC1");
            }
            if (chair.size != desk.size || chair.size != cabilnet.size) {
                throw std::invalid_argument("furnitures_map must have same size");
            }

            cv::flip(chair, chair, 0);    // (上下)翻转
            cv::flip(desk, desk, 0);    // (上下)翻转
            cv::flip(cabilnet, cabilnet, 0);    // (上下)翻转

            load_resolution = resolution;
            load_size[0] = chair.size[1] * resolution;
            load_size[1] = chair.size[0] * resolution;
            cv_chair = chair;
            cv_desk = desk;
            cv_cabilnet = cabilnet;
        }
    }

    // 家具类型
    enum class FurnitureType {
        DESK = 0x01,       // 桌子
        CHAIR = 0x02,      // 椅子
        CABILNET = 0x04,   // 柜子
        UNKNOWN = DESK | CHAIR | CABILNET,    // 未知
    };

    static FurnitureType get_furniture_by_name(std::string name) {
        if (name == "DESK") {
            return FurnitureType::DESK;
        }
        else if (name == "CHAIR") {
            return FurnitureType::CHAIR;
        }
        else if (name == "CABILNET") {
            return FurnitureType::CABILNET;
        }
        else {
            return FurnitureType::UNKNOWN;
        }
    }

    static std::string get_name_by_type(const FurnitureType& type) {
        if (type == FurnitureType::DESK) {
            return "DESK";
        }
        else if (type == FurnitureType::CHAIR) {
            return "CHAIR";
        }
        else if (type == FurnitureType::CABILNET) {
            return "CABILNET";
        }
        else {
            return "UNKNOWN";
        }
    }

    // 只需要物体尺寸, 默认地图中心为物体中心
    FurnitureMap(int id, const Eigen::Vector2d &size=load_size, double resolution=load_resolution) :
        id(id),
        quadrant(0), 
        OccupyMap(-size / 2, size / 2, resolution, PIXEL_UNKNOWN)
    {
    }

    // 获取世界坐标系下, 地图四个角的坐标, 左下, 左上, 右上, 右下
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> get_map_corners() {
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> corners;
        // 转换到世界坐标系 T_WorldSensor * T_SensorPoint
        auto pos2d = pos3d_to_2d(pos);
        corners.push_back(pos2d * Eigen::Vector2d(left_bottom[0], left_bottom[1]));
        corners.push_back(pos2d * Eigen::Vector2d(left_bottom[0], right_top[1]));
        corners.push_back(pos2d * Eigen::Vector2d(right_top[0], right_top[1]));
        corners.push_back(pos2d * Eigen::Vector2d(right_top[0], left_bottom[1]));
        return corners;
    }

    // 获取世界坐标系下, 物体占据空间四个角的坐标, 左下, 左上, 右上, 右下
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> get_corners(double addition_size=0) {
        check_type();
        cv::Size2d size;
        if (type == FurnitureMap::FurnitureType::CABILNET) {
            size = FurnitureMap::cabilnet_size;
        }
        else if (type == FurnitureMap::FurnitureType::DESK) {
            size = FurnitureMap::desk_size;
        }
        else if (type == FurnitureMap::FurnitureType::CHAIR) {
            size = FurnitureMap::chair_size;
        }
        else {
            throw std::out_of_range("Only recognized furniture has bounding size");
        }
        size.width += addition_size;
        size.height += addition_size;
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> corners;
        // 转换到世界坐标系 T_WorldSensor * T_SensorPoint
        auto pos2d = pos3d_to_2d(pos);
        corners.push_back(pos2d * Eigen::Vector2d(-size.width / 2, -size.height / 2));
        corners.push_back(pos2d * Eigen::Vector2d(-size.width / 2, size.height / 2));
        corners.push_back(pos2d * Eigen::Vector2d(size.width / 2, size.height / 2));
        corners.push_back(pos2d * Eigen::Vector2d(size.width / 2, -size.height / 2));
        return corners;
    }

    Eigen::Affine2d get_universal_pos2d() const {
        return pos3d_to_2d(get_universal_pos3d());
    }

    Eigen::Affine2d get_pos2d() const {
        return pos3d_to_2d(pos);
    }

    //将家具的参考坐标系, 变换到与世界坐标系一致
    //    柜子参考坐标系 绕自身X轴旋转90° 绕旋转后的Z轴旋转90°
    //    桌子参考坐标系 绕自身X轴旋转90° 绕旋转后的Z轴旋转90°
    //    椅子参考坐标系 绕自身Z轴旋转180°
    Eigen::Affine3d get_universal_pos3d() const {
        Eigen::Affine3d universal_pos3d;
        check_type();
        if (type == FurnitureType::DESK) {
            universal_pos3d =
                pos *
                Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(1, 0, 0)) *
                Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1));
        }
        else if (type == FurnitureType::CABILNET) {
            universal_pos3d =
                pos *
                Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(1, 0, 0)) *
                Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1));
        }
        else if (type == FurnitureType::CHAIR) {
            universal_pos3d =
                pos *
                Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0, 0, 1));
        }
        else {
            throw std::runtime_error("get_universal_pos need to determine furniture type");
        }
        return universal_pos3d;
    }

    const Eigen::Affine3d& get_pos3d() const {
        return pos;
    }

    // 更新家具位姿
    void update_pos(const Eigen::Affine3d &new_pos) {
        this->pos = new_pos;
        check_quadrant();
    }

    // 根据传感器结果更新家具地图, start的X轴正方向为传感器朝向
    void update(bool is_occupied,
        Eigen::Affine2d start, Eigen::Vector2d end = Eigen::Vector2d(),
        double min_distance = 0, double max_distance = 2, double angle = 0) {

        // 转换到子地图坐标系 T_SensorWorld * T_WorldStart
        auto pos2d = pos3d_to_2d(pos);
        start = pos2d.inverse() * start;
        end = pos2d.inverse() * end;

        // 设置激光沿线为空闲
        if (angle == 0) {
            // single line distance sensor
            Eigen::Vector2i start_pos, free_pos;
            if (min_distance > 0) {
                start_pos = this->convert_to_map_pos(start.translation() + start.rotation() * Eigen::Vector2d(min_distance, 0));
            }
            else {
                start_pos = this->convert_to_map_pos(start.translation());
            }
            if (is_occupied) {
                free_pos = this->convert_to_map_pos(end);
            }
            else {
                free_pos = this->convert_to_map_pos(start.translation() + start.rotation() * Eigen::Vector2d(max_distance, 0));
            }
            // iterate over line
            cv::LineIterator it(cv_data, cv::Point(start_pos[0], start_pos[1]), cv::Point(free_pos[0], free_pos[1]), 8, false);
            for (int i = 0; i < it.count; i++, ++it) {
                cv::Point p = it.pos();
                if (check_position(Eigen::Vector2i(p.x, p.y))) {
                    if (_get(p.x, p.y) == PIXEL_UNKNOWN) {
                        _set(p.x, p.y, PIXEL_FREE);
                    }
                }
            }
        }

        // 设置激光端点为占用
        if (is_occupied) {
            Eigen::Vector2i start_pos = this->convert_to_map_pos(start.translation());
            Eigen::Vector2i end_pos = this->convert_to_map_pos(end);
            if (this->check_position(end_pos)) {
                this->_set(end_pos[0], end_pos[1], PIXEL_OCCUPIED);
            }
        }

    }

    // 根据目前已知信息, 推测家具类型
    FurnitureType check_type() const {
        if (type != FurnitureType::DESK && type != FurnitureType::CABILNET && type != FurnitureType::CHAIR) {
            cv::Mat result;
            if (cv::countNonZero(result = cv_data | cv_desk) < static_cast<int>(cv_data.total())) {
                type = FurnitureType::DESK;
            }
			else if (cv::countNonZero(result = cv_data | cv_chair) < static_cast<int>(cv_data.total())) {
				type = FurnitureType::CHAIR;
			}
			else if (cv::countNonZero(result = cv_data | cv_cabilnet) < static_cast<int>(cv_data.total())) {
                type = FurnitureType::CABILNET;
            }
        }
        return type;
    }

    // 根据家具位置, 推测家具所属的象限
    int check_quadrant() {
        if (quadrant <= 0) {
            Eigen::Vector2d pos = get_pos2d().translation();
            Eigen::Vector2d rel_pos = pos - task2_random_left_bottom;
            Eigen::Vector2d area = task2_random_right_top - task2_random_left_bottom;
            double percent_x = rel_pos[0] / area[0];
            double percent_y = rel_pos[1] / area[1];
            if (0 <= percent_x && percent_x < 0.5) {
                if (0 <= percent_y && percent_y < 0.5) {
                    quadrant = 3;
                }
                else if (percent_y <= 1) {
                    quadrant = 2;
                }
            }
            else if (percent_x <= 1) {
                if (0 <= percent_y && percent_y < 0.5) {
                    quadrant = 4;
                }
                else if (percent_y <= 1) {
                    quadrant = 1;
                }
            }
        }
        return quadrant;
    }

	// 判断家具是否在起始区域
	bool is_in_src_area() {
		return hust::is_in_area(get_pos2d().translation(), task2_src_left_bottom, task2_src_right_top);
	}

    // 判断家具是否在目标区域
    bool is_in_dst_area() {
        for (const Eigen::Vector2d& p : get_corners()) {
            if (!hust::is_in_area(p, task2_dst_left_bottom, task2_dst_right_top)) {
                return false;
            }
        }
        return true;
    }

    // 生成rgba格式图像
    uint8_t *generate_rgba(const cv::Scalar &rgb_color = cv::Scalar(0, 0, 0)) {
        // 懒惰创建rgba图像
        if (cv_rgba.empty()) {
            cv_rgba = cv::Mat(size[1], size[0], CV_8UC4, cv::Scalar(0, 0, 0, 0));
        }
        else {
            cv_rgba = cv::Vec4b(0, 0, 0, 0);
        }
        cv::Mat map = cv_data;
        if (type == FurnitureType::DESK) {
            map = cv_desk;
        }
        else if (type == FurnitureType::CHAIR) {
            map = cv_chair;
        }
        else if (type == FurnitureType::CABILNET) {
            map = cv_cabilnet;
        }
        for (int y = 0; y < size[1]; y++) {
            for (int x = 0; x < size[0]; x++) {
                if (map.at<uint8_t>(y, x) == PIXEL_OCCUPIED) {
                    cv_rgba.at<cv::Vec4b>(y, x) = cv::Vec4b(
                        static_cast<uint8_t>(rgb_color[0]), 
                        static_cast<uint8_t>(rgb_color[1]),
                        static_cast<uint8_t>(rgb_color[2]),
                        255);
                }
            }
        }
        return cv_rgba.data;
    }

    const int id;           // 家具编号
private:
    Eigen::Affine3d pos;    // 家具位姿
    cv::Mat cv_rgba;        // 用于显示的RGBA图像
    mutable FurnitureType type = FurnitureType::UNKNOWN;     // 家具类型
    mutable int quadrant;           // 家具所属象限
};

// 记录地图中的障碍物
class Obstacle {
public:
    static const std::pair<double, double> task3_radius_range;
    static const double task3_obs_min_distance;
	static const double task3_obs_wall_min_distance;
    static const Eigen::Vector2d task3_left_bottom;
    static const Eigen::Vector2d task3_right_top;
    int id;
private:
    Eigen::Affine3d pos;
    std::pair<double, double> radius_range = std::pair<double, double>(0, 0);
    mutable bool is_task3 = false;   // 是否为任务三障碍物

public:

    Obstacle(int id) :
        id(id) 
    {
    }

    Obstacle(const Obstacle&) = default; // 允许 拷贝构造
    Obstacle& operator=(const Obstacle&) = default; // 允许 拷贝赋值

    Obstacle(Obstacle&&) = default;   // 移动构造
    Obstacle& operator=(Obstacle&& other) = default;  // 移动赋值

    void update_pos(const Eigen::Affine3d& new_pos) {
        pos = new_pos;
    }

    const Eigen::Affine3d& get_pos() const {
        return pos;
    }

    const std::pair<double, double>& get_radius_range() const {
        return radius_range;
    }

    void update(const Eigen::Vector2d& end, const Eigen::Affine3d& new_pos) {
        pos = new_pos;
        double radius = (end - OccupyMap::pos3d_to_2d(pos).translation()).norm();
        if (radius_range.first == 0 || radius < radius_range.first) {
            radius_range.first = radius;
        }
        if (radius_range.second == 0 || radius > radius_range.second) {
            radius_range.second = radius;
        }
    }

    // 根据障碍物位置与半径, 判断是否为任务三障碍
    bool check_task3() const {
        if (radius_range.first >= task3_radius_range.first &&
            radius_range.second <= task3_radius_range.second &&
            task3_left_bottom[0] <= pos.translation()[0] &&
            task3_left_bottom[1] <= pos.translation()[1] &&
            task3_right_top[0] >= pos.translation()[0] &&
            task3_right_top[1] >= pos.translation()[1]) 
        {
            is_task3 = true;
        }
        return is_task3;
    }

    static std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> get_task3_obstacles(const std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> &all_obs) {
        std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> task3_obs;
        for (const Obstacle& obs : all_obs) {
            if (obs.check_task3()) {
                task3_obs.push_back(obs);
            }
        }
        return task3_obs;
    }

    // 返回按顺序排列好的任务三障碍物, 数量可能不足5个, 表示仍未发现全部的任务三障碍(或者剩余的障碍物不满足要求)
    static std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> get_ordered_task3_obstacles(const std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> &all_obs) {
        std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> task3_obs;
        for (const Obstacle& obs : all_obs) {
            if (obs.check_task3()) {
                task3_obs.push_back(obs);
            }
        }
        if (task3_obs.empty()) {
            return task3_obs;
        }

        // 障碍物顺序判断策略
        // 按X轴坐标, 从大到小排序障碍物
		std::sort(task3_obs.begin(), task3_obs.end(), [](const Obstacle& a, const Obstacle& b) -> bool {
			return OccupyMap::pos3d_to_2d(a.get_pos()).translation()[0] > OccupyMap::pos3d_to_2d(b.get_pos()).translation()[0];
		});
        return task3_obs;
    }


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};

// 机器人全局地图
class GlobalMap : public OccupyMap
{
public:
    // 任务三最终目标位姿
    static const Eigen::Vector2d task3_dst;
    static const Eigen::Vector2d task3_dynamical_left_bottom;
    static const Eigen::Vector2d task3_dynamical_right_top;
	static const Eigen::Vector2d task3_wall_random_left_bottom;
	static const Eigen::Vector2d task3_wall_random_right_top;
    static const Eigen::Vector2d task3_wall_size;
    static const double task3_wall_min_y;
    // 任务四煤气罐半径
    static const double gas_tank_radius;
    // 任务四煤气罐目标区域
    static const Eigen::Vector2d task4_dst_left_bottom;
    static const Eigen::Vector2d task4_dst_right_top;
    // 任务五门口区域
    static const Eigen::Vector2d task5_door_left_bottom;
    static const Eigen::Vector2d task5_door_right_top;
    // 任务五煤气罐目标区域
    static const Eigen::Vector2d task5_dst_left_bottom;
    static const Eigen::Vector2d task5_dst_right_top;
public:
    // 检测到的障碍物枚举类型
    enum class ObjectType {
        UNKNOWN,    // 未知
        FURNITURE,  // 任务二家具
        GASTANK,    // 任务四煤气罐
    };

    // 任务二 家具
    mutable std::mutex mutex_furnitures;    // 读写家具数组的互斥锁, 修改像素无需加锁
    std::vector<FurnitureMap, Eigen::aligned_allocator<FurnitureMap>> furnitures;   // 记录已发现的家具

    // 任务三 障碍物
    mutable std::mutex mutex_obstacles;     // 读写任务三障碍物数组的互斥锁
    std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> obstacles;   // 记录已发现的所有障碍物
    std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> task3_obstacles;  // 记录已发现的任务三障碍物

    // 任务四 煤气罐
    mutable std::mutex mutex_gas_tank;     // 读写任务三煤气罐信息的互斥锁
    int gas_tank_id = 0;    // 煤气罐id, 0表示尚未发现煤气罐
    Eigen::Affine3d gas_tank_pos;   // 煤气罐位姿

    // 导航地图
    std::string origin_map_path;    // 预定义地图的路径

    mutable std::mutex mutex_plan_map;     // 互斥锁
    int plan_map_id = 0;
    cv::Mat plan_map;
    bool plan_need_update = true;   // 导航地图需要更新
    std::vector<int> plan_ignored_ids;
    double plan_min_radius = -1;
    double plan_max_radius = -1;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> plan_addition_lines;

    GlobalMap(const Eigen::Vector2d &corner_left_bottom, const Eigen::Vector2d &corner_right_top, double resolution, std::string furniture_map_path, std::string origin_map_path) :
        OccupyMap(corner_left_bottom, corner_right_top, resolution, PIXEL_UNKNOWN),
        origin_map_path(origin_map_path)
    {
        // 读取预先生成的场景地图
        load(origin_map_path);
        // 预先生成的地图仅保留不可进入与被占据的像素值
        cv_data.setTo(cv::Scalar(PIXEL_UNKNOWN), cv_data > PIXEL_MAY_CLS);
        // 读取预先生成的家具地图
        FurnitureMap::load_furnitures_map(furniture_map_path, resolution=0.02);
    }

    // 复位
    void reset() {
        // 复位场景地图
        load(origin_map_path);
        // 预先生成的地图仅保留不可进入与被占据的像素值
        cv_data.setTo(cv::Scalar(PIXEL_UNKNOWN), cv_data > PIXEL_MAY_CLS);

        // 复位家具
        mutex_furnitures.lock();
        furnitures.clear();
        mutex_furnitures.unlock();

        // 复位障碍物
        mutex_obstacles.lock();
        obstacles.clear();
        task3_obstacles.clear();
        mutex_obstacles.unlock();

        // 复位煤气罐
        mutex_gas_tank.lock();
        gas_tank_id = 0;
        mutex_gas_tank.unlock();

        // 需要更新导航地图
        {
            std::lock_guard<std::mutex> lk(mutex_plan_map);
            plan_need_update = true;
        }
    }

    // 获取已发现的家具列表
    std::vector<int> get_furniture_ids() const {
        std::lock_guard<std::mutex> lk(mutex_furnitures);
        std::vector<int> ids;
        for (const FurnitureMap& fur : furnitures) {
            ids.push_back(fur.id);
        }
        return ids;
    }

    // 返回已发现的家具, 并计算所属的位置 (4象限)
    std::vector<FurnitureMap*> get_furnitures_area() {
        std::lock_guard<std::mutex> lk(mutex_furnitures);
        std::vector<std::vector<FurnitureMap*>> quadrants(4);
        for (FurnitureMap& fur : furnitures) {
            int q = fur.check_quadrant();
            if (q > 0) {
				quadrants[q-1].push_back(&fur);
            }
        }
		if (quadrants[0].size() > 1 || quadrants[1].size() > 1) {
			// 1 或 2 象限有多个家具
			std::vector<FurnitureMap*> fur_12;
			fur_12.insert(fur_12.end(), quadrants[0].begin(), quadrants[0].end());
			fur_12.insert(fur_12.end(), quadrants[1].begin(), quadrants[1].end());
			if (fur_12.size() > 2) {
				// 1 2 象限共有3个家具
				// todo
			}
			else {
				// 1 2 象限共有2个家具, 按照左右顺序来
				if (fur_12[0]->get_pos2d().translation()[0] < fur_12[1]->get_pos2d().translation()[0]) {
					quadrants[1] = { fur_12[0] };
					quadrants[0] = { fur_12[1] };
				}
				else {
					quadrants[0] = { fur_12[0] };
					quadrants[1] = { fur_12[1] };
				}
			}
		}
		if (quadrants[2].size() > 1 || quadrants[3].size() > 1) {
			// 3 或 4 象限有多个家具
			std::vector<FurnitureMap*> fur_34;
			fur_34.insert(fur_34.end(), quadrants[2].begin(), quadrants[2].end());
			fur_34.insert(fur_34.end(), quadrants[3].begin(), quadrants[3].end());
			if (fur_34.size() > 2) {
				// 3 4 象限共有3个家具
				// todo
			}
			else {
				// 3 4 象限共有2个家具, 按照左右顺序来
				if (fur_34[0]->get_pos2d().translation()[0] < fur_34[1]->get_pos2d().translation()[0]) {
					quadrants[3] = { fur_34[0] };
					quadrants[2] = { fur_34[1] };
				}
				else {
					quadrants[2] = { fur_34[0] };
					quadrants[3] = { fur_34[1] };
				}
			}
		}
		std::vector<FurnitureMap*> ret(4, NULL);
		for (int i = 0; i < 4; i++) {
			if (quadrants[i].size() >= 1) {
				ret[i] = quadrants[i][0];
			}
		}
        return ret;
    }

    // 更新家具位姿
    // 若仍未发现对应id的家具, 则抛出 out_of_range
    void update_furniture_pos(int id, const Eigen::Affine3d& new_pos) {
        std::lock_guard<std::mutex> lk(mutex_furnitures);
        for (FurnitureMap& fur : furnitures) {
            if (fur.id == id) {

                // 判断是否正在搬运该家具
                if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), id) == plan_ignored_ids.end()) {
                    // 位姿发生变化后需要更新导航地图
                    Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(fur.get_pos3d());
                    Eigen::Affine2d new_pos2d = OccupyMap::pos3d_to_2d(new_pos);
                    if ((pos2d.translation() - new_pos2d.translation()).norm() >= 0.01 ||
                        std::abs(hust::diff_angle(pos2d, new_pos2d)) >= 5 / 180.0f * M_PI
                        ) {
                        // 需要更新导航地图
                        std::lock_guard<std::mutex> lk(mutex_plan_map);
                        plan_need_update = true;
                        hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because furniture " + std::to_string(id) + " pos changed");
                    }
                }

                fur.update_pos(new_pos);
                return;
            }
        }
        throw std::out_of_range("Did not found furniture id");
    }

    std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> get_ordered_task3_obs() {
        std::lock_guard<std::mutex> lk(mutex_obstacles);
        return Obstacle::get_ordered_task3_obstacles(obstacles);
    }

    // 获取煤气罐id
    // 若仍未发现煤气罐, 则抛出 out_of_range
    int get_gas_tank_id() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        return gas_tank_id;
    }

    // 获取煤气罐位姿
    // 若仍未发现煤气罐, 则抛出 out_of_range
    const Eigen::Affine3d& get_gas_tank_pos() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        return gas_tank_pos;
    }

    // 判断煤气罐是否在任务四目标区域
    bool is_gas_task_in_task4_dst_area() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        Eigen::Affine2d pos2d = pos3d_to_2d(gas_tank_pos);
        // 判断圆上四个点是否都在区域内
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(gas_tank_radius, 0), task4_dst_left_bottom, task4_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(-gas_tank_radius, 0), task4_dst_left_bottom, task4_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(0, gas_tank_radius), task4_dst_left_bottom, task4_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(0, -gas_tank_radius), task4_dst_left_bottom, task4_dst_right_top)) {
            return false;
        }
        return true;
    }

    // 判断煤气罐是否在任务五目标区域
    bool is_gas_task_in_task5_dst_area() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        Eigen::Affine2d pos2d = pos3d_to_2d(gas_tank_pos);
        // 判断圆上四个点是否都在区域内
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(gas_tank_radius, 0), task5_dst_left_bottom, task5_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(-gas_tank_radius, 0), task5_dst_left_bottom, task5_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(0, gas_tank_radius), task5_dst_left_bottom, task5_dst_right_top)) {
            return false;
        }
        if (!hust::is_in_area(pos2d * Eigen::Vector2d(0, -gas_tank_radius), task5_dst_left_bottom, task5_dst_right_top)) {
            return false;
        }
        return true;
    }

    // 更新煤气罐位姿
    // 若仍未发现煤气罐, 则抛出 out_of_range
    void update_gas_tank_pos(const Eigen::Affine3d& new_pos) {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }

        // 判断是否正在搬运煤气罐
        if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), gas_tank_id) == plan_ignored_ids.end()) {
            // 位姿发生变化后需要更新导航地图
            Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(gas_tank_pos);
            Eigen::Affine2d new_pos2d = OccupyMap::pos3d_to_2d(new_pos);
            if ((pos2d.translation() - new_pos2d.translation()).norm() >= 0.01 ||
                std::abs(hust::diff_angle(pos2d, new_pos2d)) >= 5 / 180.0f * M_PI
                ) {
                // 需要更新导航地图
                std::lock_guard<std::mutex> lk(mutex_plan_map);
                plan_need_update = true;
                hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because gas_tank pos changed");
            }
        }

        gas_tank_pos = new_pos;
    }

    // 保存地图信息, 因坐标原因保存出的地图为上下翻转的
    // file_path: 文件名(不含后缀)
    // file_extension: 文件后缀(不含点)
    void save_all(std::string file_path, std::string file_extension) const {
        this->save(file_path + "." + file_extension);
        this->mutex_furnitures.lock();
        for (const FurnitureMap& fur : furnitures) {
            fur.save(file_path + "_" + std::to_string(fur.id) + "." + file_extension);
        }
        this->mutex_furnitures.unlock();
    }

    // 根据传感器结果更新地图
    // start: 传感器位姿, 其X轴正方向为传感器朝向
    // end: 扫描到的端点位置
    // min_distance: 最小扫描距离
    // max_distance: 最大扫描距离
    // angle: 传感器的扫描角度
    // object_id: 扫描到的物体编号, 若为任务2中的家具则编号>0
    // object_pos: 扫描到的物体位姿
	void update(bool is_occupied,
        const Eigen::Affine2d &start, const Eigen::Vector2d &end = Eigen::Vector2d(),
        double min_distance = 0, double max_distance = 2, double angle = 0, 
        ObjectType object_type = ObjectType::UNKNOWN, int object_id = -1, const Eigen::Affine3d &object_pos = Eigen::Affine3d()) {

        if (angle == 0) {
            // single line distance sensor
            Eigen::Vector2i start_pos, free_pos;
            if (min_distance > 0) {
                start_pos = this->convert_to_map_pos(start.translation() + start.rotation() * Eigen::Vector2d(min_distance, 0));
            }
            else {
                start_pos = this->convert_to_map_pos(start.translation());
            }
            if (is_occupied) {
                free_pos = this->convert_to_map_pos(end);
            }
            else {
                free_pos = this->convert_to_map_pos(start.translation() + start.rotation() * Eigen::Vector2d(max_distance, 0));
            }
            // iterate over line
            cv::LineIterator it(cv_data, cv::Point(start_pos[0], start_pos[1]), cv::Point(free_pos[0], free_pos[1]), 8, false);
            for (int i = 0; i < it.count; i++, ++it) {
                cv::Point p = it.pos();
                if (check_position(Eigen::Vector2i(p.x, p.y))) {
                    if (_get(p.x, p.y) == PIXEL_UNKNOWN) {
                        _set(p.x, p.y, PIXEL_FREE);
                    }
                }
            }
        }
        else {
            // area distance sensor
            // set disk range free
            double free_distance;
            if (is_occupied) {
                free_distance = (start.translation() - end).norm();
            }
            else {
                free_distance = max_distance;
            }
            if (free_distance - min_distance >= resolution * 3 && angle * M_PI / 180 * min_distance >= resolution * 3) {
                // OpenCV draw courter line, and flood fill with free
                /*
                Eigen::Transform<double, 2, Eigen::Affine> left = start * Eigen::Rotation2Dd(angle * M_PI / 180 / 2);
                Eigen::Vector2d line_left_begin = left * Eigen::Vector2d(min_distance, 0);
                Eigen::Vector2d line_left_end = left * Eigen::Vector2d(free_distance, 0);

                Eigen::Transform<double, 2, Eigen::Affine> right = start * Eigen::Rotation2Dd(-angle * M_PI / 180 / 2);
                Eigen::Vector2d line_right_begin = right * Eigen::Vector2d(min_distance, 0);
                Eigen::Vector2d line_right_end = right * Eigen::Vector2d(free_distance, 0);

                cv::line(cv_data, cv::Point2i(std::round(line_left_begin[0]), std::round(line_left_begin[1])),
                cv::Point2i(std::round(line_left_end[0]), std::round(line_left_end[1])), cv::Scalar(0));
                cv::line(cv_data, cv::Point2i(std::round(line_right_begin[0]), std::round(line_right_end[1])),
                cv::Point2i(std::round(line_left_end[0]), std::round(line_left_end[1])), cv::Scalar(0));*/
            }
        }

        // set end point occupied
        if (is_occupied) {
            Eigen::Vector2i start_pos = this->convert_to_map_pos(start.translation());
            Eigen::Vector2i end_pos = this->convert_to_map_pos(end);
            if (object_type == ObjectType::FURNITURE) {
                // 检测到的是任务二 家具
                // 检查是否曾经检测到该家具
                FurnitureMap * p_furniture = NULL;
                mutex_furnitures.lock();
                for (FurnitureMap& fur : furnitures) {
                    if (fur.id == object_id) {
                        p_furniture = &fur;
                        break;
                    }
                }
                // 更新家具信息, 需要加锁
                if (p_furniture == NULL) {
                    // 为新检测到的家具
                    furnitures.emplace_back(object_id);
                    p_furniture = &furnitures[furnitures.size() - 1];
                    // 导航地图需要更新
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because new furniture detected");
                }
                // 更新家具位姿
                p_furniture->update_pos(object_pos);
                mutex_furnitures.unlock();
                // 更新家具子地图
                p_furniture->update(is_occupied, start, end, min_distance, max_distance, angle);
                // 检查家具类型
                mutex_furnitures.lock();
                p_furniture->check_type();
                mutex_furnitures.unlock();
            }
            else if (object_type == ObjectType::GASTANK) {
                // 监测到的是任务四 煤气罐
                mutex_gas_tank.lock();
                if (gas_tank_id == 0) {
                    gas_tank_id = object_id;
                    gas_tank_pos = object_pos;
                    // 导航地图需要更新
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because gas_tank detected");
                }
                mutex_gas_tank.unlock();
            }
            else {
                // 未知障碍物
                Obstacle * p_obs = NULL;
                // 检查是否曾经发现该障碍物
                for (Obstacle& obs : obstacles) {
                    if (obs.id == object_id) {
                        p_obs = &obs;
                        break;
                    }
                }
                if (p_obs == NULL) {
                    // 这是新发现的未知障碍物, 记录
                    obstacles.emplace_back(object_id);
                    p_obs = &obstacles.back();
                }
                p_obs->update(end, object_pos);

                // 更新任务三障碍物列表
                auto new_task3_obstacles = Obstacle::get_task3_obstacles(obstacles);
                if (!std::equal(
                    task3_obstacles.begin(), task3_obstacles.end(),
                    new_task3_obstacles.begin(), new_task3_obstacles.end(),
                    [] (const Obstacle& a, const Obstacle& b) -> bool {
                        return a.id == b.id;
                    }
                )) {
                    task3_obstacles = new_task3_obstacles;
                    // 需要更新导航地图
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because new task3 obs detected");
                }

                // 更新到全局地图
                if (this->check_position(end_pos)) {
                    uint8_t prev_value = _get(end_pos[0], end_pos[1]);
                    this->_set(end_pos[0], end_pos[1], PIXEL_OCCUPIED);
                    if (prev_value > PIXEL_LIMITED && !p_obs->check_task3() && 
                        end[0] >= task3_wall_random_left_bottom[0] && end[1] >= task3_wall_random_left_bottom[1] &&
                        end[0] <= task3_wall_random_right_top[0] && end[1] <= task3_wall_random_right_top[1]
                    ) {
                        // 发现了动态墙体, 更新地图
                        // 将可能的墙体置为PIXEL_LIMITED
                        size_t wall_w = static_cast<int>(std::round(task3_wall_size[0] / resolution));
                        size_t wall_h = static_cast<int>(std::round(task3_wall_size[1] / resolution));
                        double min_y = std::min(task3_wall_min_y, end[1]);
                        Eigen::Vector2i wall_left_bottom = convert_to_map_pos(Eigen::Vector2d(end[0] - task3_wall_size[0], min_y));
                        for (int x = wall_left_bottom[0]; x <= wall_left_bottom[0] + wall_w; x++) {
                            for (int y = wall_left_bottom[1]; y <= wall_left_bottom[1] + wall_h; y++) {
                                Eigen::Vector2i p(x, y);
                                if (check_position(p)) {
                                    if (_get(p) > PIXEL_LIMITED) {
                                        _set(p, PIXEL_LIMITED);
                                    }
                                }
                            }
                        }

                        std::lock_guard<std::mutex> lk(mutex_plan_map);
                        plan_need_update = true;
                        std::ostringstream ss;
                        ss << "Plan map (id=" << plan_map_id << ") need update! Because unexpected obs (" << (int)prev_value << " -> " << (int)PIXEL_OCCUPIED;
                        hust::log(ss.str());
                    }
                }
                 
            }
        }
        else {
            // 更新家具子地图
            for (FurnitureMap& fur : furnitures) {
                fur.update(false, start, end, min_distance, max_distance, angle);
            }
        }
    }

    bool set_plan_addition_lines(
        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& addition_lines = {},
        double line_pos_tol = 0.001
    ) {
        // 额外的线段障碍
        if (addition_lines.size() % 2 != 0) {
            throw std::invalid_argument("size of addition_lines must divede by 2");
        }

        std::lock_guard<std::mutex> lk(mutex_plan_map);

        bool need_update = false;
        if (plan_addition_lines.size() != addition_lines.size()) {
            plan_addition_lines = addition_lines;
            need_update = true;
        }
        else {
            for (size_t i = 0; i < addition_lines.size(); i++) {
                if ((plan_addition_lines[i] - addition_lines[i]).norm() > line_pos_tol) {
                    plan_addition_lines = addition_lines;
                    need_update = true;
                    break;
                }
            }
        }

        if (need_update) {
            // 导航地图需要更新
            plan_need_update = true;
            hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because addition_lines changed");
        }

        return need_update;
    }

    bool set_plan_map_ignores(const std::vector<int> &ignore_ids) {
        std::lock_guard<std::mutex> lk(mutex_plan_map);

        bool need_update = false;
        if (plan_ignored_ids != ignore_ids) {
            plan_ignored_ids = ignore_ids;
            need_update = true;
        }
        if (need_update) {
            // 导航地图需要更新
            plan_need_update = true;
            std::ostringstream ss;
            ss << "Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because ignore_ids changed";
            ss << "New ignores={";
            for (const int& id : plan_ignored_ids) {
                ss << id << ", ";
            }
            ss << "}";
            hust::log(ss.str());
        }
        return need_update;
    }

    // 设置导航地图的参数
    bool set_plan_map_params(
        double car_min_radius = -1,
        double car_max_radius = -1, 
        double radius_tol = 0.001
    ) {
        std::lock_guard<std::mutex> lk(mutex_plan_map);

        bool need_update = false;
        if (car_min_radius >= 0 && std::abs(car_min_radius - plan_min_radius) > radius_tol) {
            plan_min_radius = car_min_radius;
            need_update = true;
        }
        if (car_max_radius >= 0 && std::abs(car_max_radius - plan_max_radius) > radius_tol) {
            plan_max_radius = car_max_radius;
            need_update = true;
        }

        if (need_update) {
            // 导航地图需要更新
            plan_need_update = true;
            std::ostringstream ss;
            ss << "Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because car_radius changed";
            ss << "radius=(" << car_min_radius << ", " << car_max_radius << ")";
            hust::log(ss.str());
        }

        return need_update;
    }

    bool is_newest_plan_map(int map_id) {
        std::lock_guard<std::mutex> lk(mutex_plan_map);
        if (map_id != plan_map_id) {
            return false;
        }
        else if (plan_need_update) {
            return false; 
        }
        else {
            return true;
        }
    }

    // 生成导航地图, 包含安全距离限制
    // (互斥锁保护)
    // 参数: 可以忽略的物体id, 如正在操作的物体(家具, 煤气罐)
    // 重绘条件:
    // 1. 需要绘制的家具出现变化
    // 2. 发现了新的任务三障碍物
    // 3. todo 发现了出现在自由区域的新障碍物
    std::pair<int, cv::Mat> generate_plan_map() {
        std::lock_guard<std::mutex> lk(mutex_plan_map);

        // 检查是否需要重新生成导航地图
        if (!plan_need_update) {
            return std::pair<int, cv::Mat>(plan_map_id, plan_map);
        }
        // 统计本次需要绘制的物体

        // 重新生成导航地图
        cv::Mat new_map = cv_data.clone();
        // 放置家具障碍
        for (FurnitureMap& fur : furnitures) {
            // 检查是否需要排除
            if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), fur.id) == plan_ignored_ids.end()) {
                if (fur.check_type() == FurnitureMap::FurnitureType::UNKNOWN) {
                    // 家具类型未知, 按检测到的被占据点绘制障碍
                    Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(fur.get_pos3d());
                    // 进行旋转变换
                    cv::Mat rotated;
                    cv::Mat cv_affine = cv::getRotationMatrix2D(
                        cv::Point(fur.size[0] / 2, fur.size[1] / 2) ,
                        -Eigen::Rotation2Dd(pos2d.rotation()).angle() / M_PI * 180,
                        1
                    );
                    cv::warpAffine(fur.cv_data, rotated, cv_affine, fur.cv_data.size(), 
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, PIXEL_FREE);
                    //cv::imwrite("C:/Program Files/Robot/luaFile/rotated.png", rotated);
                    // 绘制到导航地图上
                    Eigen::Vector2i center = convert_to_map_pos(pos2d.translation());
                    cv::Point left_bottom(
                        center[0] - fur.size[0] / 2,
                        center[1] - fur.size[1] / 2
                    );
                    cv::Point right_top(
                        center[0] + fur.size[0] / 2,
                        center[1] + fur.size[1] / 2
                    );
                    cv::Mat mask = rotated <= PIXEL_LIMITED;
                    //cv::imwrite("C:/Program Files/Robot/luaFile/mask.png", mask);
                    plan_map(cv::Rect(left_bottom, right_top)).setTo(0x00, mask);
                }
                else {
                    // 家具类型已知, 按外接矩形绘制障碍
                    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> corners = fur.get_corners(resolution);    // 需要在长宽上添加额外的尺寸
                    // 转换为整型坐标点
                    std::vector<cv::Point> cv_points;
                    for (Eigen::Vector2d& pd : corners) {
                        auto point = convert_to_map_pos(pd);
                        cv_points.push_back(cv::Point(point[0], point[1]));
                    }
                    // 绘制外接矩形
                    for (int i = 0; i < 4; i++) {
                        cv::LineIterator it(new_map, cv_points[i], cv_points[(i + 1) % 4], 8, false);
                        for (int i = 0; i < it.count; i++, ++it) {
                            cv::Point p = it.pos();
                            if (check_position(Eigen::Vector2i(p.x, p.y))) {
                                if (new_map.at<uint8_t>(p) > PIXEL_LIMITED) {
                                    new_map.at<uint8_t>(p) = PIXEL_LIMITED;
                                }
                            }
                        }
                    }
                }

            }
        }
        // 放置任务三障碍物
        double task3_obs_radius = (Obstacle::task3_radius_range.first + Obstacle::task3_radius_range.second) / 2;
        for (const Obstacle& obs : Obstacle::get_task3_obstacles(obstacles)) {
            Eigen::Vector2d pos2d = pos3d_to_2d(obs.get_pos()).translation();
            auto point = convert_to_map_pos(pos2d);
            cv::Point center(point[0], point[1]);
            int r = static_cast<int>(std::round(task3_obs_radius / resolution + 0.5));    // 需要在半径上添加额外的尺寸
            cv::circle(new_map, center, r, cv::Scalar(PIXEL_OCCUPIED), 1, 4);
        }
        // 放置任务四煤气罐障碍物
        if (gas_tank_id != 0) {
            Eigen::Vector2d pos2d = pos3d_to_2d(gas_tank_pos).translation();
            auto point = convert_to_map_pos(pos2d);
            cv::Point center(point[0], point[1]);
            int r = static_cast<int>(std::round(gas_tank_radius / resolution + 0.5));    // 需要在半径上添加额外的尺寸
            cv::circle(new_map, center, r, cv::Scalar(PIXEL_OCCUPIED), 1, 4);
        }
        // 放置额外的线段障碍
        for (size_t i = 0; i < plan_addition_lines.size(); i += 2) {
            Eigen::Vector2i p0 = convert_to_map_pos(plan_addition_lines[i]);
            Eigen::Vector2i p1 = convert_to_map_pos(plan_addition_lines[i+1]);
            cv::line(new_map, cv::Point(p0[0], p0[1]), cv::Point(p1[0], p1[1]), PIXEL_OCCUPIED);
        }

        // 根据机器人的最大半径, 将有可能发生碰撞位置的像素值身为PIXEL_MAY_CLS
        if (plan_min_radius > 0 && plan_max_radius > plan_min_radius) {
            int max_radius = static_cast<int>(std::round(plan_max_radius / resolution - 0.5));   // 安全半径
            int min_radius = static_cast<int>(std::round(plan_min_radius / resolution - 0.5));   // 安全半径
            //uint8_t color_max = PIXEL_UNKNOWN - 1;
            uint8_t color_max = PIXEL_MAY_CLS;
            uint8_t color_min = PIXEL_MAY_CLS;
            for (int radius = max_radius; radius > min_radius; radius--) {
                cv::Mat occupied_area = new_map <= PIXEL_LIMITED;  // 被占据的区域为255
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2, radius * 2));  // 模板为圆形
                cv::Mat eroded_area;
                cv::dilate(occupied_area, eroded_area, kernel);
                cv::Mat diff_area = occupied_area ^ eroded_area;    // 腐蚀改变的区域
                double color_percent = 1.0 * (radius - min_radius) / (max_radius - min_radius);
                new_map.setTo(cv::Scalar(static_cast<uint8_t>(color_min + color_percent * (color_max - color_min))), diff_area); // 腐蚀改变的区域设为PIXEL_MAY_CLS
            }
        }
        // 为保证最大安全距离, 对<=PIXEL_LIMITED的区域进行腐蚀, 腐蚀改变的区域设为PIXEL_NOPLAN
        if (plan_min_radius > 0) {
            cv::Mat occupied_area = new_map <= PIXEL_LIMITED;  // 被占据的区域为255
            int safe_radius = static_cast<int>(std::round(plan_min_radius / resolution - 0.5));   // 安全半径
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(safe_radius * 2, safe_radius * 2));  // 模板为圆形
            cv::Mat eroded_area;
            cv::dilate(occupied_area, eroded_area, kernel);
            cv::Mat diff_area = occupied_area ^ eroded_area;    // 腐蚀改变的区域
            new_map.setTo(cv::Scalar(PIXEL_NOT_PLAN), diff_area); // 腐蚀改变的区域设为PIXEL_NOPLAN
        }

        plan_map_id++;
        plan_map = new_map;
        plan_need_update = false;
        hust::log("[MAP] Plan map updated! (id=" + std::to_string(plan_map_id) + ")");
        return std::pair<int, cv::Mat>(plan_map_id, plan_map);
    }
};
