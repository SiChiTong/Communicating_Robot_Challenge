#pragma once

#include <string>
#include <algorithm>

#include "sol.hpp"

class hust {
public:
    static bool log(const std::string& str);

    static bool log_status(const std::string& str);

    static Eigen::Affine3d convert_to_Affine3d(sol::object &obj) {
        // check type
        if (!obj.is<sol::table>()) {
            throw std::invalid_argument("convert_to_Affine3d() param must be lua table");
        }
        auto table = obj.as<sol::table>();
        // check if legal
        if (table.size() != 12) {
            throw std::invalid_argument("Transform matrix size must be 12");
        }
        bool is_legal = true;
        Eigen::Matrix<double, 3, 4> matrix;
        table.for_each([&is_legal, &matrix](sol::object key, sol::object value) {
            if (key.is<double>() && value.is<double>()) {
                int index = static_cast<int>(key.as<double>()) - 1;
                if (index >= 12 || index < 0) {
                    throw std::invalid_argument("Transform matrix index must in [0, 12)");
                }
                int row = index / 4;
                int col = index % 4;
                matrix(row, col) = value.as<double>();
            }
            else {
                is_legal = false;
            }
        });
        if (!is_legal) {
            throw std::invalid_argument("Bad transform matrix");
        }
        return Eigen::Affine3d(matrix);
    }

    template <typename _type = double>
    static std::vector<_type> convert_to_vector(sol::object &obj, int max_size = -1) {
        // check type
        if (!obj.is<sol::table>()) {
            throw std::invalid_argument("convert_to_vector() param must be lua table");
        }
        auto table = obj.as<sol::table>();
        // get size
        int length;
        if (max_size == -1) {
            length = table.size();
        }
        else if (max_size < (int)table.size()) {
            length = max_size;
        }
        else {
            length = table.size();
        }
        std::vector<_type> ret;
        for (int i = 0; i < length; i++) {
            sol::object value = table[i + 1];
            if (!value.is<_type>()) {
                throw std::invalid_argument("convert_to_vector() index " + std::to_string(i + 1) + " is not _type");
            }
            ret.push_back(value.as<_type>());
        }
        return ret;
    }

    template <typename _type = double, int _length>
    static std::array<_type, _length> convert_to_array(sol::object &obj) {
        std::vector<_type> li = convert_to_vector(obj, _length);
        if (li.size() < _length) {
            throw std::invalid_argument("convert_to_array() size=" + std::to_string(li.size()) + ", " + std::to_string(_length) + " expected");
        }
        std::array<_type, _length> arr;
        std::copy(li.begin(), li.end(), arr.begin());
        return arr;
    }

    // a + diff = b, diff in [-pi, pi)
    static double sub_2pi(double a, double b) {
        b -= a;
        while (b < -M_PI) {
            b += 2 * M_PI;
        }
        while (b > M_PI) {
            b -= 2 * M_PI;
        }
        return b;
    }

    static double diff_angle(const Eigen::Affine2d& a, const Eigen::Affine2d& b) {
        return sub_2pi(
            Eigen::Rotation2Dd(a.rotation()).angle(),
            Eigen::Rotation2Dd(b.rotation()).angle()
        );
    }

    static double get_angle(const Eigen::Affine2d& pos) {
        return Eigen::Rotation2Dd(pos.rotation()).angle();
    }

    static bool is_in_area(
        const Eigen::Vector2d& p,
        const Eigen::Vector2d& left_bottom,
        const Eigen::Vector2d& right_top
    ) {
        if (left_bottom[0] <= p[0] && p[0] <= right_top[0] &&
            left_bottom[1] <= p[1] && p[1] <= right_top[1]
        ) {
            return true;
        }
        else {
            return false;
        }
    }

    static Eigen::Affine2d xyw_to_affine2d(const std::array<double, 3>& pos) {
        return Eigen::Translation2d(pos[0], pos[1]) * Eigen::Rotation2Dd(pos[2]);
    }

    static std::array<double, 3> affine2D_to_xyw(const Eigen::Affine2d& pos) {
        std::array<double, 3> xyw;
        xyw[0] = pos.translation()[0];
        xyw[1] = pos.translation()[1];
        xyw[2] = get_angle(pos);
        return xyw;
    }
};

