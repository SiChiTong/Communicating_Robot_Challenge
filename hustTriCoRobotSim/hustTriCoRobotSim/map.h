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

// ������ͼ��
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
    MapBase(const MapBase&) = delete; // ��ֹ ��������
    MapBase& operator=(const MapBase&) = delete; // ��ֹ ������ֵ

    MapBase(MapBase&&) = default;   // �ƶ�����
    MapBase& operator=(MapBase&& other) = default;  // �ƶ���ֵ


    /* ���ݶ�ȡ */
    const uint8_t *get_data() const {
        return this->cv_data.data;
    }

    // ����Խ����ĵ�ͼ���ݶ�ȡ
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

    // ����Խ����ĵ�ͼ����д��
    void set(const Eigen::Vector2i &pos, uint8_t value) {
        if (!this->check_position(pos)) {
            std::stringstream ss;
            ss << "Map range: ([0, " << size[0] << "], [0, " << size[1] << "]), given index: (" << pos[0] << ", " << pos[1] << ")";
            throw std::out_of_range(ss.str());
        }
        this->_set(pos, value);
    }

    // ��double������������ ��ȡ��Ӧ��int���͵�ͼ����
    Eigen::Vector2i convert_to_map_pos(const Eigen::Vector2d &pos_in_world) const {
        Eigen::Vector2d pos = (pos_in_world - left_bottom) / resolution;
        return Eigen::Vector2i(static_cast<int>(std::round(pos[0])), static_cast<int>(std::round(pos[1])));
    }

    // ��int���͵�ͼ���� ת��Ϊ double������������
    Eigen::Vector2d convert_to_world_pos(const Eigen::Vector2i &map_pos) const {
        return map_pos.cast<double>() * resolution + left_bottom;
    }

    // ���Խ��
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

    // �����ͼ ��Ҫдȫ·�� �ļ��� �ļ���׺
    // ��������ԭ��, ����Ͷ�ȡʱ��Ҫ���·�ת
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

    // ��ȡ��ͼ
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


    /* ���º���ֱ�Ӷ�д����, �����Խ�� */
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

// ռ��դ���ͼ
class OccupyMap : public MapBase {

public:
    static const uint8_t PIXEL_OCCUPIED =   0x00;   // ��ʵ��ռ��
    static const uint8_t PIXEL_LIMITED =    0x50;   // �����ϰ���ռ����ε�ԭ��, �����ɽ��������
    static const uint8_t PIXEL_NOT_PLAN =   0x80;   // ���ڳ���ߴ�Ͱ�ȫ��������ľ��Բ��ɵ�������
    static const uint8_t PIXEL_MAY_CLS =    0xA0;   // �ڸ���������ʻ, ���ڳ���ߴ�, ���ܷ�����ײ
    static const uint8_t PIXEL_UNKNOWN =    0xE0;   // δ֪
    static const uint8_t PIXEL_FREE =       0xFF;   // ����

    // ʹ�û��๹�캯��
    using MapBase::MapBase;

    // ��ά��תͶӰ����ά
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

// �Ҿߵ��ӵ�ͼ
class FurnitureMap : public OccupyMap
{
public:
    static const cv::Size2d chair_size;
    static const cv::Size2d desk_size;
    static const cv::Size2d cabilnet_size;
    // ������Ҿ�ԭʼλ�÷�Χ
    static const Eigen::Vector2d task2_src_left_bottom;
    static const Eigen::Vector2d task2_src_right_top;
    // ������Ҿ�Ŀ��λ�÷�Χ
    static const Eigen::Vector2d task2_dst_left_bottom;
    static const Eigen::Vector2d task2_dst_right_top;
    // ������Ҿ��������λ�÷�Χ
    static const Eigen::Vector2d task2_random_left_bottom;
    static const Eigen::Vector2d task2_random_right_top;
private:
    // ��¼Ԥ�ȶ�ȡ�ļҾߵ�ͼ
    static cv::Mat cv_chair;
    static cv::Mat cv_desk;
    static cv::Mat cv_cabilnet;
    static Eigen::Vector2d load_size;
    static double load_resolution;
public:
    // ��ȡԤ�ȼ�¼�ļҾߵ�ͼ
    // �Ҷ�ֵ 0: ʵ��ռ��, 255: ����, ����ֵ[1, 254]: ���ɽ���
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

            cv::flip(chair, chair, 0);    // (����)��ת
            cv::flip(desk, desk, 0);    // (����)��ת
            cv::flip(cabilnet, cabilnet, 0);    // (����)��ת

            load_resolution = resolution;
            load_size[0] = chair.size[1] * resolution;
            load_size[1] = chair.size[0] * resolution;
            cv_chair = chair;
            cv_desk = desk;
            cv_cabilnet = cabilnet;
        }
    }

    // �Ҿ�����
    enum class FurnitureType {
        DESK = 0x01,       // ����
        CHAIR = 0x02,      // ����
        CABILNET = 0x04,   // ����
        UNKNOWN = DESK | CHAIR | CABILNET,    // δ֪
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

    // ֻ��Ҫ����ߴ�, Ĭ�ϵ�ͼ����Ϊ��������
    FurnitureMap(int id, const Eigen::Vector2d &size=load_size, double resolution=load_resolution) :
        id(id),
        quadrant(0), 
        OccupyMap(-size / 2, size / 2, resolution, PIXEL_UNKNOWN)
    {
    }

    // ��ȡ��������ϵ��, ��ͼ�ĸ��ǵ�����, ����, ����, ����, ����
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> get_map_corners() {
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> corners;
        // ת������������ϵ T_WorldSensor * T_SensorPoint
        auto pos2d = pos3d_to_2d(pos);
        corners.push_back(pos2d * Eigen::Vector2d(left_bottom[0], left_bottom[1]));
        corners.push_back(pos2d * Eigen::Vector2d(left_bottom[0], right_top[1]));
        corners.push_back(pos2d * Eigen::Vector2d(right_top[0], right_top[1]));
        corners.push_back(pos2d * Eigen::Vector2d(right_top[0], left_bottom[1]));
        return corners;
    }

    // ��ȡ��������ϵ��, ����ռ�ݿռ��ĸ��ǵ�����, ����, ����, ����, ����
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
        // ת������������ϵ T_WorldSensor * T_SensorPoint
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

    //���ҾߵĲο�����ϵ, �任������������ϵһ��
    //    ���Ӳο�����ϵ ������X����ת90�� ����ת���Z����ת90��
    //    ���Ӳο�����ϵ ������X����ת90�� ����ת���Z����ת90��
    //    ���Ӳο�����ϵ ������Z����ת180��
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

    // ���¼Ҿ�λ��
    void update_pos(const Eigen::Affine3d &new_pos) {
        this->pos = new_pos;
        check_quadrant();
    }

    // ���ݴ�����������¼Ҿߵ�ͼ, start��X��������Ϊ����������
    void update(bool is_occupied,
        Eigen::Affine2d start, Eigen::Vector2d end = Eigen::Vector2d(),
        double min_distance = 0, double max_distance = 2, double angle = 0) {

        // ת�����ӵ�ͼ����ϵ T_SensorWorld * T_WorldStart
        auto pos2d = pos3d_to_2d(pos);
        start = pos2d.inverse() * start;
        end = pos2d.inverse() * end;

        // ���ü�������Ϊ����
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

        // ���ü���˵�Ϊռ��
        if (is_occupied) {
            Eigen::Vector2i start_pos = this->convert_to_map_pos(start.translation());
            Eigen::Vector2i end_pos = this->convert_to_map_pos(end);
            if (this->check_position(end_pos)) {
                this->_set(end_pos[0], end_pos[1], PIXEL_OCCUPIED);
            }
        }

    }

    // ����Ŀǰ��֪��Ϣ, �Ʋ�Ҿ�����
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

    // ���ݼҾ�λ��, �Ʋ�Ҿ�����������
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

	// �жϼҾ��Ƿ�����ʼ����
	bool is_in_src_area() {
		return hust::is_in_area(get_pos2d().translation(), task2_src_left_bottom, task2_src_right_top);
	}

    // �жϼҾ��Ƿ���Ŀ������
    bool is_in_dst_area() {
        for (const Eigen::Vector2d& p : get_corners()) {
            if (!hust::is_in_area(p, task2_dst_left_bottom, task2_dst_right_top)) {
                return false;
            }
        }
        return true;
    }

    // ����rgba��ʽͼ��
    uint8_t *generate_rgba(const cv::Scalar &rgb_color = cv::Scalar(0, 0, 0)) {
        // ���贴��rgbaͼ��
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

    const int id;           // �Ҿ߱��
private:
    Eigen::Affine3d pos;    // �Ҿ�λ��
    cv::Mat cv_rgba;        // ������ʾ��RGBAͼ��
    mutable FurnitureType type = FurnitureType::UNKNOWN;     // �Ҿ�����
    mutable int quadrant;           // �Ҿ���������
};

// ��¼��ͼ�е��ϰ���
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
    mutable bool is_task3 = false;   // �Ƿ�Ϊ�������ϰ���

public:

    Obstacle(int id) :
        id(id) 
    {
    }

    Obstacle(const Obstacle&) = default; // ���� ��������
    Obstacle& operator=(const Obstacle&) = default; // ���� ������ֵ

    Obstacle(Obstacle&&) = default;   // �ƶ�����
    Obstacle& operator=(Obstacle&& other) = default;  // �ƶ���ֵ

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

    // �����ϰ���λ����뾶, �ж��Ƿ�Ϊ�������ϰ�
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

    // ���ذ�˳�����кõ��������ϰ���, �������ܲ���5��, ��ʾ��δ����ȫ�����������ϰ�(����ʣ����ϰ��ﲻ����Ҫ��)
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

        // �ϰ���˳���жϲ���
        // ��X������, �Ӵ�С�����ϰ���
		std::sort(task3_obs.begin(), task3_obs.end(), [](const Obstacle& a, const Obstacle& b) -> bool {
			return OccupyMap::pos3d_to_2d(a.get_pos()).translation()[0] > OccupyMap::pos3d_to_2d(b.get_pos()).translation()[0];
		});
        return task3_obs;
    }


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

};

// ������ȫ�ֵ�ͼ
class GlobalMap : public OccupyMap
{
public:
    // ����������Ŀ��λ��
    static const Eigen::Vector2d task3_dst;
    static const Eigen::Vector2d task3_dynamical_left_bottom;
    static const Eigen::Vector2d task3_dynamical_right_top;
	static const Eigen::Vector2d task3_wall_random_left_bottom;
	static const Eigen::Vector2d task3_wall_random_right_top;
    static const Eigen::Vector2d task3_wall_size;
    static const double task3_wall_min_y;
    // ������ú���ް뾶
    static const double gas_tank_radius;
    // ������ú����Ŀ������
    static const Eigen::Vector2d task4_dst_left_bottom;
    static const Eigen::Vector2d task4_dst_right_top;
    // �������ſ�����
    static const Eigen::Vector2d task5_door_left_bottom;
    static const Eigen::Vector2d task5_door_right_top;
    // ������ú����Ŀ������
    static const Eigen::Vector2d task5_dst_left_bottom;
    static const Eigen::Vector2d task5_dst_right_top;
public:
    // ��⵽���ϰ���ö������
    enum class ObjectType {
        UNKNOWN,    // δ֪
        FURNITURE,  // ������Ҿ�
        GASTANK,    // ������ú����
    };

    // ����� �Ҿ�
    mutable std::mutex mutex_furnitures;    // ��д�Ҿ�����Ļ�����, �޸������������
    std::vector<FurnitureMap, Eigen::aligned_allocator<FurnitureMap>> furnitures;   // ��¼�ѷ��ֵļҾ�

    // ������ �ϰ���
    mutable std::mutex mutex_obstacles;     // ��д�������ϰ�������Ļ�����
    std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> obstacles;   // ��¼�ѷ��ֵ������ϰ���
    std::vector<Obstacle, Eigen::aligned_allocator<Obstacle>> task3_obstacles;  // ��¼�ѷ��ֵ��������ϰ���

    // ������ ú����
    mutable std::mutex mutex_gas_tank;     // ��д������ú������Ϣ�Ļ�����
    int gas_tank_id = 0;    // ú����id, 0��ʾ��δ����ú����
    Eigen::Affine3d gas_tank_pos;   // ú����λ��

    // ������ͼ
    std::string origin_map_path;    // Ԥ�����ͼ��·��

    mutable std::mutex mutex_plan_map;     // ������
    int plan_map_id = 0;
    cv::Mat plan_map;
    bool plan_need_update = true;   // ������ͼ��Ҫ����
    std::vector<int> plan_ignored_ids;
    double plan_min_radius = -1;
    double plan_max_radius = -1;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> plan_addition_lines;

    GlobalMap(const Eigen::Vector2d &corner_left_bottom, const Eigen::Vector2d &corner_right_top, double resolution, std::string furniture_map_path, std::string origin_map_path) :
        OccupyMap(corner_left_bottom, corner_right_top, resolution, PIXEL_UNKNOWN),
        origin_map_path(origin_map_path)
    {
        // ��ȡԤ�����ɵĳ�����ͼ
        load(origin_map_path);
        // Ԥ�����ɵĵ�ͼ���������ɽ����뱻ռ�ݵ�����ֵ
        cv_data.setTo(cv::Scalar(PIXEL_UNKNOWN), cv_data > PIXEL_MAY_CLS);
        // ��ȡԤ�����ɵļҾߵ�ͼ
        FurnitureMap::load_furnitures_map(furniture_map_path, resolution=0.02);
    }

    // ��λ
    void reset() {
        // ��λ������ͼ
        load(origin_map_path);
        // Ԥ�����ɵĵ�ͼ���������ɽ����뱻ռ�ݵ�����ֵ
        cv_data.setTo(cv::Scalar(PIXEL_UNKNOWN), cv_data > PIXEL_MAY_CLS);

        // ��λ�Ҿ�
        mutex_furnitures.lock();
        furnitures.clear();
        mutex_furnitures.unlock();

        // ��λ�ϰ���
        mutex_obstacles.lock();
        obstacles.clear();
        task3_obstacles.clear();
        mutex_obstacles.unlock();

        // ��λú����
        mutex_gas_tank.lock();
        gas_tank_id = 0;
        mutex_gas_tank.unlock();

        // ��Ҫ���µ�����ͼ
        {
            std::lock_guard<std::mutex> lk(mutex_plan_map);
            plan_need_update = true;
        }
    }

    // ��ȡ�ѷ��ֵļҾ��б�
    std::vector<int> get_furniture_ids() const {
        std::lock_guard<std::mutex> lk(mutex_furnitures);
        std::vector<int> ids;
        for (const FurnitureMap& fur : furnitures) {
            ids.push_back(fur.id);
        }
        return ids;
    }

    // �����ѷ��ֵļҾ�, ������������λ�� (4����)
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
			// 1 �� 2 �����ж���Ҿ�
			std::vector<FurnitureMap*> fur_12;
			fur_12.insert(fur_12.end(), quadrants[0].begin(), quadrants[0].end());
			fur_12.insert(fur_12.end(), quadrants[1].begin(), quadrants[1].end());
			if (fur_12.size() > 2) {
				// 1 2 ���޹���3���Ҿ�
				// todo
			}
			else {
				// 1 2 ���޹���2���Ҿ�, ��������˳����
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
			// 3 �� 4 �����ж���Ҿ�
			std::vector<FurnitureMap*> fur_34;
			fur_34.insert(fur_34.end(), quadrants[2].begin(), quadrants[2].end());
			fur_34.insert(fur_34.end(), quadrants[3].begin(), quadrants[3].end());
			if (fur_34.size() > 2) {
				// 3 4 ���޹���3���Ҿ�
				// todo
			}
			else {
				// 3 4 ���޹���2���Ҿ�, ��������˳����
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

    // ���¼Ҿ�λ��
    // ����δ���ֶ�Ӧid�ļҾ�, ���׳� out_of_range
    void update_furniture_pos(int id, const Eigen::Affine3d& new_pos) {
        std::lock_guard<std::mutex> lk(mutex_furnitures);
        for (FurnitureMap& fur : furnitures) {
            if (fur.id == id) {

                // �ж��Ƿ����ڰ��˸üҾ�
                if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), id) == plan_ignored_ids.end()) {
                    // λ�˷����仯����Ҫ���µ�����ͼ
                    Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(fur.get_pos3d());
                    Eigen::Affine2d new_pos2d = OccupyMap::pos3d_to_2d(new_pos);
                    if ((pos2d.translation() - new_pos2d.translation()).norm() >= 0.01 ||
                        std::abs(hust::diff_angle(pos2d, new_pos2d)) >= 5 / 180.0f * M_PI
                        ) {
                        // ��Ҫ���µ�����ͼ
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

    // ��ȡú����id
    // ����δ����ú����, ���׳� out_of_range
    int get_gas_tank_id() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        return gas_tank_id;
    }

    // ��ȡú����λ��
    // ����δ����ú����, ���׳� out_of_range
    const Eigen::Affine3d& get_gas_tank_pos() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        return gas_tank_pos;
    }

    // �ж�ú�����Ƿ���������Ŀ������
    bool is_gas_task_in_task4_dst_area() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        Eigen::Affine2d pos2d = pos3d_to_2d(gas_tank_pos);
        // �ж�Բ���ĸ����Ƿ���������
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

    // �ж�ú�����Ƿ���������Ŀ������
    bool is_gas_task_in_task5_dst_area() const {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }
        Eigen::Affine2d pos2d = pos3d_to_2d(gas_tank_pos);
        // �ж�Բ���ĸ����Ƿ���������
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

    // ����ú����λ��
    // ����δ����ú����, ���׳� out_of_range
    void update_gas_tank_pos(const Eigen::Affine3d& new_pos) {
        std::lock_guard<std::mutex> lk(mutex_gas_tank);
        if (gas_tank_id == 0) {
            throw std::out_of_range("Did not found gas tank");
        }

        // �ж��Ƿ����ڰ���ú����
        if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), gas_tank_id) == plan_ignored_ids.end()) {
            // λ�˷����仯����Ҫ���µ�����ͼ
            Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(gas_tank_pos);
            Eigen::Affine2d new_pos2d = OccupyMap::pos3d_to_2d(new_pos);
            if ((pos2d.translation() - new_pos2d.translation()).norm() >= 0.01 ||
                std::abs(hust::diff_angle(pos2d, new_pos2d)) >= 5 / 180.0f * M_PI
                ) {
                // ��Ҫ���µ�����ͼ
                std::lock_guard<std::mutex> lk(mutex_plan_map);
                plan_need_update = true;
                hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because gas_tank pos changed");
            }
        }

        gas_tank_pos = new_pos;
    }

    // �����ͼ��Ϣ, ������ԭ�򱣴���ĵ�ͼΪ���·�ת��
    // file_path: �ļ���(������׺)
    // file_extension: �ļ���׺(������)
    void save_all(std::string file_path, std::string file_extension) const {
        this->save(file_path + "." + file_extension);
        this->mutex_furnitures.lock();
        for (const FurnitureMap& fur : furnitures) {
            fur.save(file_path + "_" + std::to_string(fur.id) + "." + file_extension);
        }
        this->mutex_furnitures.unlock();
    }

    // ���ݴ�����������µ�ͼ
    // start: ������λ��, ��X��������Ϊ����������
    // end: ɨ�赽�Ķ˵�λ��
    // min_distance: ��Сɨ�����
    // max_distance: ���ɨ�����
    // angle: ��������ɨ��Ƕ�
    // object_id: ɨ�赽��������, ��Ϊ����2�еļҾ�����>0
    // object_pos: ɨ�赽������λ��
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
                // ��⵽��������� �Ҿ�
                // ����Ƿ�������⵽�üҾ�
                FurnitureMap * p_furniture = NULL;
                mutex_furnitures.lock();
                for (FurnitureMap& fur : furnitures) {
                    if (fur.id == object_id) {
                        p_furniture = &fur;
                        break;
                    }
                }
                // ���¼Ҿ���Ϣ, ��Ҫ����
                if (p_furniture == NULL) {
                    // Ϊ�¼�⵽�ļҾ�
                    furnitures.emplace_back(object_id);
                    p_furniture = &furnitures[furnitures.size() - 1];
                    // ������ͼ��Ҫ����
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because new furniture detected");
                }
                // ���¼Ҿ�λ��
                p_furniture->update_pos(object_pos);
                mutex_furnitures.unlock();
                // ���¼Ҿ��ӵ�ͼ
                p_furniture->update(is_occupied, start, end, min_distance, max_distance, angle);
                // ���Ҿ�����
                mutex_furnitures.lock();
                p_furniture->check_type();
                mutex_furnitures.unlock();
            }
            else if (object_type == ObjectType::GASTANK) {
                // ��⵽���������� ú����
                mutex_gas_tank.lock();
                if (gas_tank_id == 0) {
                    gas_tank_id = object_id;
                    gas_tank_pos = object_pos;
                    // ������ͼ��Ҫ����
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because gas_tank detected");
                }
                mutex_gas_tank.unlock();
            }
            else {
                // δ֪�ϰ���
                Obstacle * p_obs = NULL;
                // ����Ƿ��������ָ��ϰ���
                for (Obstacle& obs : obstacles) {
                    if (obs.id == object_id) {
                        p_obs = &obs;
                        break;
                    }
                }
                if (p_obs == NULL) {
                    // �����·��ֵ�δ֪�ϰ���, ��¼
                    obstacles.emplace_back(object_id);
                    p_obs = &obstacles.back();
                }
                p_obs->update(end, object_pos);

                // �����������ϰ����б�
                auto new_task3_obstacles = Obstacle::get_task3_obstacles(obstacles);
                if (!std::equal(
                    task3_obstacles.begin(), task3_obstacles.end(),
                    new_task3_obstacles.begin(), new_task3_obstacles.end(),
                    [] (const Obstacle& a, const Obstacle& b) -> bool {
                        return a.id == b.id;
                    }
                )) {
                    task3_obstacles = new_task3_obstacles;
                    // ��Ҫ���µ�����ͼ
                    std::lock_guard<std::mutex> lk(mutex_plan_map);
                    plan_need_update = true;
                    hust::log("Plan map (id=" + std::to_string(plan_map_id) + ") need update! Because new task3 obs detected");
                }

                // ���µ�ȫ�ֵ�ͼ
                if (this->check_position(end_pos)) {
                    uint8_t prev_value = _get(end_pos[0], end_pos[1]);
                    this->_set(end_pos[0], end_pos[1], PIXEL_OCCUPIED);
                    if (prev_value > PIXEL_LIMITED && !p_obs->check_task3() && 
                        end[0] >= task3_wall_random_left_bottom[0] && end[1] >= task3_wall_random_left_bottom[1] &&
                        end[0] <= task3_wall_random_right_top[0] && end[1] <= task3_wall_random_right_top[1]
                    ) {
                        // �����˶�̬ǽ��, ���µ�ͼ
                        // �����ܵ�ǽ����ΪPIXEL_LIMITED
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
            // ���¼Ҿ��ӵ�ͼ
            for (FurnitureMap& fur : furnitures) {
                fur.update(false, start, end, min_distance, max_distance, angle);
            }
        }
    }

    bool set_plan_addition_lines(
        const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& addition_lines = {},
        double line_pos_tol = 0.001
    ) {
        // ������߶��ϰ�
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
            // ������ͼ��Ҫ����
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
            // ������ͼ��Ҫ����
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

    // ���õ�����ͼ�Ĳ���
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
            // ������ͼ��Ҫ����
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

    // ���ɵ�����ͼ, ������ȫ��������
    // (����������)
    // ����: ���Ժ��Ե�����id, �����ڲ���������(�Ҿ�, ú����)
    // �ػ�����:
    // 1. ��Ҫ���ƵļҾ߳��ֱ仯
    // 2. �������µ��������ϰ���
    // 3. todo �����˳�����������������ϰ���
    std::pair<int, cv::Mat> generate_plan_map() {
        std::lock_guard<std::mutex> lk(mutex_plan_map);

        // ����Ƿ���Ҫ�������ɵ�����ͼ
        if (!plan_need_update) {
            return std::pair<int, cv::Mat>(plan_map_id, plan_map);
        }
        // ͳ�Ʊ�����Ҫ���Ƶ�����

        // �������ɵ�����ͼ
        cv::Mat new_map = cv_data.clone();
        // ���üҾ��ϰ�
        for (FurnitureMap& fur : furnitures) {
            // ����Ƿ���Ҫ�ų�
            if (std::find(plan_ignored_ids.begin(), plan_ignored_ids.end(), fur.id) == plan_ignored_ids.end()) {
                if (fur.check_type() == FurnitureMap::FurnitureType::UNKNOWN) {
                    // �Ҿ�����δ֪, ����⵽�ı�ռ�ݵ�����ϰ�
                    Eigen::Affine2d pos2d = OccupyMap::pos3d_to_2d(fur.get_pos3d());
                    // ������ת�任
                    cv::Mat rotated;
                    cv::Mat cv_affine = cv::getRotationMatrix2D(
                        cv::Point(fur.size[0] / 2, fur.size[1] / 2) ,
                        -Eigen::Rotation2Dd(pos2d.rotation()).angle() / M_PI * 180,
                        1
                    );
                    cv::warpAffine(fur.cv_data, rotated, cv_affine, fur.cv_data.size(), 
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, PIXEL_FREE);
                    //cv::imwrite("C:/Program Files/Robot/luaFile/rotated.png", rotated);
                    // ���Ƶ�������ͼ��
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
                    // �Ҿ�������֪, ����Ӿ��λ����ϰ�
                    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> corners = fur.get_corners(resolution);    // ��Ҫ�ڳ�������Ӷ���ĳߴ�
                    // ת��Ϊ���������
                    std::vector<cv::Point> cv_points;
                    for (Eigen::Vector2d& pd : corners) {
                        auto point = convert_to_map_pos(pd);
                        cv_points.push_back(cv::Point(point[0], point[1]));
                    }
                    // ������Ӿ���
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
        // �����������ϰ���
        double task3_obs_radius = (Obstacle::task3_radius_range.first + Obstacle::task3_radius_range.second) / 2;
        for (const Obstacle& obs : Obstacle::get_task3_obstacles(obstacles)) {
            Eigen::Vector2d pos2d = pos3d_to_2d(obs.get_pos()).translation();
            auto point = convert_to_map_pos(pos2d);
            cv::Point center(point[0], point[1]);
            int r = static_cast<int>(std::round(task3_obs_radius / resolution + 0.5));    // ��Ҫ�ڰ뾶����Ӷ���ĳߴ�
            cv::circle(new_map, center, r, cv::Scalar(PIXEL_OCCUPIED), 1, 4);
        }
        // ����������ú�����ϰ���
        if (gas_tank_id != 0) {
            Eigen::Vector2d pos2d = pos3d_to_2d(gas_tank_pos).translation();
            auto point = convert_to_map_pos(pos2d);
            cv::Point center(point[0], point[1]);
            int r = static_cast<int>(std::round(gas_tank_radius / resolution + 0.5));    // ��Ҫ�ڰ뾶����Ӷ���ĳߴ�
            cv::circle(new_map, center, r, cv::Scalar(PIXEL_OCCUPIED), 1, 4);
        }
        // ���ö�����߶��ϰ�
        for (size_t i = 0; i < plan_addition_lines.size(); i += 2) {
            Eigen::Vector2i p0 = convert_to_map_pos(plan_addition_lines[i]);
            Eigen::Vector2i p1 = convert_to_map_pos(plan_addition_lines[i+1]);
            cv::line(new_map, cv::Point(p0[0], p0[1]), cv::Point(p1[0], p1[1]), PIXEL_OCCUPIED);
        }

        // ���ݻ����˵����뾶, ���п��ܷ�����ײλ�õ�����ֵ��ΪPIXEL_MAY_CLS
        if (plan_min_radius > 0 && plan_max_radius > plan_min_radius) {
            int max_radius = static_cast<int>(std::round(plan_max_radius / resolution - 0.5));   // ��ȫ�뾶
            int min_radius = static_cast<int>(std::round(plan_min_radius / resolution - 0.5));   // ��ȫ�뾶
            //uint8_t color_max = PIXEL_UNKNOWN - 1;
            uint8_t color_max = PIXEL_MAY_CLS;
            uint8_t color_min = PIXEL_MAY_CLS;
            for (int radius = max_radius; radius > min_radius; radius--) {
                cv::Mat occupied_area = new_map <= PIXEL_LIMITED;  // ��ռ�ݵ�����Ϊ255
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(radius * 2, radius * 2));  // ģ��ΪԲ��
                cv::Mat eroded_area;
                cv::dilate(occupied_area, eroded_area, kernel);
                cv::Mat diff_area = occupied_area ^ eroded_area;    // ��ʴ�ı������
                double color_percent = 1.0 * (radius - min_radius) / (max_radius - min_radius);
                new_map.setTo(cv::Scalar(static_cast<uint8_t>(color_min + color_percent * (color_max - color_min))), diff_area); // ��ʴ�ı��������ΪPIXEL_MAY_CLS
            }
        }
        // Ϊ��֤���ȫ����, ��<=PIXEL_LIMITED��������и�ʴ, ��ʴ�ı��������ΪPIXEL_NOPLAN
        if (plan_min_radius > 0) {
            cv::Mat occupied_area = new_map <= PIXEL_LIMITED;  // ��ռ�ݵ�����Ϊ255
            int safe_radius = static_cast<int>(std::round(plan_min_radius / resolution - 0.5));   // ��ȫ�뾶
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(safe_radius * 2, safe_radius * 2));  // ģ��ΪԲ��
            cv::Mat eroded_area;
            cv::dilate(occupied_area, eroded_area, kernel);
            cv::Mat diff_area = occupied_area ^ eroded_area;    // ��ʴ�ı������
            new_map.setTo(cv::Scalar(PIXEL_NOT_PLAN), diff_area); // ��ʴ�ı��������ΪPIXEL_NOPLAN
        }

        plan_map_id++;
        plan_map = new_map;
        plan_need_update = false;
        hust::log("[MAP] Plan map updated! (id=" + std::to_string(plan_map_id) + ")");
        return std::pair<int, cv::Mat>(plan_map_id, plan_map);
    }
};
