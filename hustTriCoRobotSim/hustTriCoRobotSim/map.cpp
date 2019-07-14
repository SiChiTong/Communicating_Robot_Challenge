#include "map.h"

cv::Mat FurnitureMap::cv_chair;
cv::Mat FurnitureMap::cv_desk;
cv::Mat FurnitureMap::cv_cabilnet;
Eigen::Vector2d FurnitureMap::load_size;
double FurnitureMap::load_resolution;

const cv::Size2d FurnitureMap::chair_size(5.8387e-01, 6.3047e-01);
const cv::Size2d FurnitureMap::desk_size(5.2121e-01, 1.2575e+00);
const cv::Size2d FurnitureMap::cabilnet_size(4.5067e-01, 8.5320e-01);
// 任务二家具起始区域
const Eigen::Vector2d FurnitureMap::task2_src_left_bottom(-3.264, -4.636);
const Eigen::Vector2d FurnitureMap::task2_src_right_top(-0.264, -2.636);
// 任务二家具目标区域
const Eigen::Vector2d FurnitureMap::task2_dst_left_bottom(-3.264, -8.325);
const Eigen::Vector2d FurnitureMap::task2_dst_right_top(0.736, -6.325);
// 任务二家具随机生成区域
const Eigen::Vector2d FurnitureMap::task2_random_left_bottom(-2.984, -4.636);
const Eigen::Vector2d FurnitureMap::task2_random_right_top(-0.264, -2.636);
// 任务三目标位置
const Eigen::Vector2d GlobalMap::task3_dst(-7.614, -7.836);

//const std::pair<double, double> Obstacle::task3_radius_range(0.118, 0.12);
const std::pair<double, double> Obstacle::task3_radius_range(0.11, 0.13);

const double Obstacle::task3_obs_min_distance = 0.88;
const double Obstacle::task3_obs_wall_min_distance = 0.74;

const Eigen::Vector2d Obstacle::task3_left_bottom(-7.414, -8.325);
const Eigen::Vector2d Obstacle::task3_right_top(-1.839, -4.614);

const Eigen::Vector2d GlobalMap::task3_dynamical_left_bottom(-8.789, -7.725);
const Eigen::Vector2d GlobalMap::task3_dynamical_right_top(-1.839, -4.614);

// 任务三墙体随机区域
const Eigen::Vector2d GlobalMap::task3_wall_random_left_bottom(-8.814, -7.311);
const Eigen::Vector2d GlobalMap::task3_wall_random_right_top(-7.464, -6.286);

const Eigen::Vector2d GlobalMap::task3_wall_size(7.714 - 7.489, 56 * 0.02);
const double GlobalMap::task3_wall_min_y = -7.311;

const double GlobalMap::gas_tank_radius = 0.18;
// 任务四煤气罐目标区域
const Eigen::Vector2d GlobalMap::task4_dst_left_bottom(-12.585, -6.762);
const Eigen::Vector2d GlobalMap::task4_dst_right_top(-11.835, -6.012);

// 任务五门口区域
const Eigen::Vector2d GlobalMap::task5_door_left_bottom(-12.785, -9.537);
const Eigen::Vector2d GlobalMap::task5_door_right_top(-11.135, -9.312);

// 任务五煤气罐目标区域
const Eigen::Vector2d GlobalMap::task5_dst_left_bottom(-6.372, -14.784);
const Eigen::Vector2d GlobalMap::task5_dst_right_top(-4.862, -13.247);

