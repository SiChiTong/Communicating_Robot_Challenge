#pragma once
#include <thread>
#include <mutex>
#include <string>
#include <vector>
#include <iterator>
#include <map>

#include <Eigen/Geometry>
#include <pangolin/pangolin.h>

#include "map.h"
#include "plan.h"

template<typename Out>
static void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

class DebugGUI {
private:
    std::thread gui_thread;

    GlobalMap *map = NULL;

    std::mutex mutex_planner;
    Planner *planner = NULL;

    cv::Mat plan_map;
    GLuint map_texture = 0;                 // 全局地图对应纹理
    std::map<int, GLuint> submap_texture;   // 子地图对应纹理
    std::function<void(void)> map_save_callback;    // 保存地图按钮回调

    std::mutex mutex_data; // 控制数据互斥锁
    std::vector<std::string> lua_cmds;
    std::vector<std::string> buffer_status_log;
    std::vector<std::string> status_log;
    std::vector<std::string> debug_log = std::vector<std::string>(100, "");
    int debug_log_index = 0;
    Eigen::Vector2d car_size;
    Eigen::Transform<double, 2, Eigen::Affine> car_pos;

public:
    DebugGUI() {
        reset_control();
    }

    void reset_control() {
        std::lock_guard<std::mutex> lock(mutex_data);
        lua_cmds.clear();
    }

    std::vector<std::string> &&get_cmds() {
        std::lock_guard<std::mutex> lock(mutex_data);
        return std::move(this->lua_cmds);
    }

    void set_map(GlobalMap *map) {
        this->map = map;
    }

    void set_planner(Planner *planner) {
        std::lock_guard<std::mutex> lk(mutex_planner);
        this->planner = planner;
    }

    /* 以下日志函数 线程安全 */
    void clear_log_status() {
        std::lock_guard<std::mutex> lock(mutex_data);
        status_log = buffer_status_log;
        this->buffer_status_log.clear();
    }
    void log_status(const std::string &str) {
        std::lock_guard<std::mutex> lock(mutex_data);
        auto lines = split(str, '\n');
        for (const std::string& line : lines) {
            if (!line.empty()) {
                buffer_status_log.push_back(line);
            }
        }
    }
    void clear_log() {
        std::lock_guard<std::mutex> lock(mutex_data);
        for (std::string& line : debug_log) {
            line = "";
        }
        debug_log_index = 0;
    }
    void log(const std::string &str) {
        std::lock_guard<std::mutex> lock(mutex_data);
        auto lines = split(str, '\n');
        for (const std::string& line : lines) {
            if (!line.empty()) {
                debug_log[debug_log_index] = line;
                debug_log_index = (debug_log_index + 1) % debug_log.size();
            }
        }
    }

    void set_car_size(Eigen::Vector2d car_size) {
        this->car_size = car_size;
    }

    void set_car_position(Eigen::Affine2d car_pos) {
        this->car_pos = car_pos;
    }

    void set_map_save_callback(const std::function<void(void)> &map_save_callback) {
        this->map_save_callback = map_save_callback;
    }

    void run_in_thread() {
        if (!gui_thread.joinable()) {
            gui_thread = std::thread(std::bind(&DebugGUI::main_func, this));
        }
    }

    void run() {
        run_in_thread();
        gui_thread.join();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
private:

    static std::string generate_lua_matrix(Eigen::Transform<double, 3, Eigen::Affine> transform) {
        std::ostringstream ss;
        ss << "{";
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                ss << transform.matrix()(r, c) << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }

    // 绘制坐标轴
    static void draw_axis(Eigen::Affine3d pos, double size, std::string name="") {
        Eigen::Affine3f posf = pos.cast<float>();
        // Draw XYZ Axis
        glColor4f(1, 0, 0, 1);
        Eigen::Vector3f axis_x = (pos * Eigen::Vector3d(size, 0, 0)).cast<float>();
        pangolin::glDrawLine(
            posf.translation()[0], posf.translation()[1], posf.translation()[2],
            axis_x[0],  axis_x[1], axis_x[2]);
        glColor4f(0, 1, 0, 1);
        Eigen::Vector3f axis_y = (pos * Eigen::Vector3d(0, size, 0)).cast<float>();
        pangolin::glDrawLine(
            posf.translation()[0], posf.translation()[1], posf.translation()[2],
            axis_y[0], axis_y[1], axis_y[2]);
        glColor4f(0, 0, 1, 1);
        Eigen::Vector3f axis_z = (pos * Eigen::Vector3d(0, 0, size)).cast<float>();
        pangolin::glDrawLine(
            posf.translation()[0], posf.translation()[1], posf.translation()[2],
            axis_z[0], axis_z[1], axis_z[2]);
        // draw label
        glColor3f(0, 0, 0);
        pangolin::GlFont::I().Text("X").Draw(axis_x[0], axis_x[1], axis_x[2]);
        pangolin::GlFont::I().Text("Y").Draw(axis_y[0], axis_y[1], axis_y[2]);
        pangolin::GlFont::I().Text("Z").Draw(axis_z[0], axis_z[1], axis_z[2]);
        pangolin::GlFont::I().Text(name.c_str()).Draw(posf.translation()[0], posf.translation()[1], posf.translation()[2]);
    }

    void main_func() {
        const int WINDOW_WIDTH = 1024;
        const int WINDOW_HEIGHT = 768;
        const int PANEL_WIDTH = 180;

        // 创建窗口
        pangolin::CreateWindowAndBind("华中科技大学 自动化学院 智能机器人实验室", WINDOW_WIDTH, WINDOW_HEIGHT);
        // 三维鼠标操作需要启用 GL_DEPTH_TEST
        glEnable(GL_DEPTH_TEST);

        // 定义相机渲染对象 (用于观察/浏览场景)
        pangolin::OpenGlRenderState s_cam(
            // 计算投影矩阵 4x4 
            // 窗口像素大小: 640x480
            // 窗口视角范围: 420x420
            // 原点位置: 320x240
            // Z轴距离范围: 0.2 - 100
            pangolin::ProjectionMatrix(WINDOW_WIDTH, WINDOW_HEIGHT, 420, 420, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2, 0.1, 200),
            // 计算相机位姿变换矩阵 4x4 (观察矩阵, 由位置+目标+上向量定义)
            // 相机位于 (-2, -2, -2) 相机朝向 (0, 0, 0) 上向量为 Y轴
            pangolin::ModelViewLookAt(-1, -1, 1, 0, 0, 0, pangolin::AxisZ)
        );

        // 添加 OpenGL 视窗到窗口, 并提供三维浏览操作
        pangolin::View& d_cam = pangolin::CreateDisplay()
			//.SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), pangolin::Attach::ReversePix(PANEL_WIDTH), static_cast<double>(-WINDOW_WIDTH) / WINDOW_HEIGHT)
			.SetBounds(0.0, 1.0, pangolin::Attach::Pix(0), pangolin::Attach::ReversePix(0), static_cast<double>(-WINDOW_WIDTH) / WINDOW_HEIGHT)
            .SetHandler(new pangolin::Handler3D(s_cam, pangolin::AxisZ));   // Z轴固定朝上

        // 添加一个面板, 命名为ui
        //pangolin::CreatePanel("ui")
        //    // 设定面板位置 参数: 底部 顶部 左侧 右侧, 参数范围为[0, 1]
        //    .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));

        pangolin::Var<bool> buttom_map_save("ui.save_map", false);

        pangolin::Var<double> vx("ui.vel_x", 0, -5, 5);
        pangolin::Var<double> vy("ui.vel_y", 0, -5, 5);
        pangolin::Var<double> vw("ui.vel_w", 0, -180, 180);

        pangolin::Var<bool> button_f("ui.move_forward", false);
        pangolin::Var<bool> button_b("ui.move_backward", false);
        pangolin::Var<bool> button_l("ui.move_left", false);
        pangolin::Var<bool> button_r("ui.move_right", false);
        pangolin::Var<bool> button_tl("ui.turn_left", false);
        pangolin::Var<bool> button_tr("ui.turn_right", false);
        pangolin::Var<bool> button_stop("ui.stop", false);

        pangolin::RegisterKeyPressCallback('w', pangolin::SetVarFunctor<bool>("ui.move_forward", true));
        pangolin::RegisterKeyPressCallback('s', pangolin::SetVarFunctor<bool>("ui.move_backward", true));
        pangolin::RegisterKeyPressCallback('q', pangolin::SetVarFunctor<bool>("ui.move_left", true));
        pangolin::RegisterKeyPressCallback('e', pangolin::SetVarFunctor<bool>("ui.move_right", true));
        pangolin::RegisterKeyPressCallback('a', pangolin::SetVarFunctor<bool>("ui.turn_left", true));
        pangolin::RegisterKeyPressCallback('d', pangolin::SetVarFunctor<bool>("ui.turn_right", true));
        pangolin::RegisterKeyPressCallback(' ', pangolin::SetVarFunctor<bool>("ui.stop", true));

        pangolin::Var<bool> button_arm_tu("ui.trans_up", false);
        pangolin::Var<bool> button_arm_td("ui.trans_down", false);
        pangolin::Var<bool> button_arm_tl("ui.trans_left", false);
        pangolin::Var<bool> button_arm_tr("ui.trans_right", false);
        pangolin::Var<bool> button_arm_tf("ui.trans_forward", false);
        pangolin::Var<bool> button_arm_tb("ui.trans_backward", false);

        pangolin::Var<bool> button_arm_rxcw("ui.rotate_x_cw", false);
        pangolin::Var<bool> button_arm_rxccw("ui.rotate_x_ccw", false);
        pangolin::Var<bool> button_arm_rycw("ui.rotate_y_cw", false);
        pangolin::Var<bool> button_arm_ryccw("ui.rotate_y_ccw", false);
        pangolin::Var<bool> button_arm_rzcw("ui.rotate_z_cw", false);
        pangolin::Var<bool> button_arm_rzccw("ui.rotate_z_ccw", false);

        // 添加一个面板, 命名为arm
        //pangolin::CreatePanel("arm")
        //    // 设定面板位置 参数: 底部 顶部 左侧 右侧, 参数范围为[0, 1]
        //    .SetBounds(0.0, 1.0, pangolin::Attach::ReversePix(PANEL_WIDTH), 1.0);

        pangolin::Var<int> button_l_lr_r("arm.L_LR_R", 0, -1, 1);
        pangolin::Var<int> button_arm_tstep("arm.trans_step", 10, 0, 100);
        pangolin::Var<int> button_arm_rstep("arm.rotate_step", 5, 0, 15);
        for (int i = 0; i < 10; i++) {
            pangolin::Var<int>("arm.joint_" + std::to_string(i + 1), 0, -1, 1);
        }

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        while (!pangolin::ShouldQuit())
        {

            if (pangolin::Pushed(buttom_map_save) && this->map_save_callback) {
                this->map_save_callback();
            }

            mutex_data.lock();
            double vel_linear = 0.6;
            double vel_rad = 75 * M_PI / 180;
            if (pangolin::Pushed(button_f)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {" << vel_linear << ", 0, 0})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_b)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {-" << vel_linear << ", 0, 0})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_l)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {0, " << vel_linear << ", 0})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_r)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {0, -" << vel_linear << ", 0})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_tl)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {0, 0, " << vel_rad << "})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_tr)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {0, 0, -" << vel_rad << "})";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_stop)) {
                std::stringstream ss;
                ss << "hustSetBaseTargetPosVel(nil, {0, 0, 0})";
                lua_cmds.push_back(ss.str());
            }

            // 机械臂电机控制, 位移部分
            if (pangolin::Pushed(button_arm_tu)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(0, 0, button_arm_tstep / 1000.0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_td)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(0, 0, -button_arm_tstep / 1000.0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_tl)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(0, button_arm_tstep / 1000.0, 0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_tr)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(0, -button_arm_tstep / 1000.0, 0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_tf)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(button_arm_tstep / 1000.0, 0, 0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_tb)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>((Eigen::Translation3d(-button_arm_tstep / 1000.0, 0, 0)))) << ")";
                lua_cmds.push_back(ss.str());
            }
            // 机械臂电机控制, 旋转部分
            if (pangolin::Pushed(button_arm_rxcw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(button_arm_rstep * M_PI / 180, Eigen::Vector3d(1, 0, 0)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_rxccw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(-button_arm_rstep * M_PI / 180, Eigen::Vector3d(1, 0, 0)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_rycw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(button_arm_rstep * M_PI / 180, Eigen::Vector3d(0, 1, 0)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_ryccw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(-button_arm_rstep * M_PI / 180, Eigen::Vector3d(0, 1, 0)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_rzcw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(button_arm_rstep * M_PI / 180, Eigen::Vector3d(0, 0, 1)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }
            if (pangolin::Pushed(button_arm_rzccw)) {
                std::stringstream ss;
                ss << "hustArmTransRel(" << button_l_lr_r << ", " << generate_lua_matrix(static_cast<Eigen::Transform<double, 3, Eigen::Affine>>(Eigen::AngleAxisd(-button_arm_rstep * M_PI / 180, Eigen::Vector3d(0, 0, 1)))) << ", 'end_pos-base_rot')";
                lua_cmds.push_back(ss.str());
            }

            // 机械臂关节控制
            for (int i = 0; i < 10; i++) {
                pangolin::Var<int> value("arm.joint_" + std::to_string(i + 1), 0, -1, 1);
                double pos_diff;
                if (i < 8) {
                    // 旋转关节
                    pos_diff = button_arm_rstep * M_PI / 180;
                }
                else {
                    // 位移关节
                    pos_diff = button_arm_tstep / 1000.0;
                }
                if (value > 0) {
                    pos_diff = pos_diff;
                    value = 0;
                }
                else if (value < 0) {
                    pos_diff = -pos_diff;
                    value = 0;
                }
                else {
                    pos_diff = 0;
                }
                if (pos_diff != 0) {
                    std::ostringstream ss;
                    ss << "hustJointPosRel(" << button_l_lr_r << ", " << i + 1 << ", " << pos_diff << ")";
                    lua_cmds.push_back(ss.str());
                }
            }

            mutex_data.unlock();


            // Clear screen and activate view to render into
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);

            // Render OpenGL Object

            // Draw XYZ Axis
            pangolin::glDrawAxis(1);
            glColor3f(1, 1, 1);
            pangolin::GlFont::I().Text("X").Draw(1, 0, 0);
            pangolin::GlFont::I().Text("Y").Draw(0, 1, 0);
            pangolin::GlFont::I().Text("Z").Draw(0, 0, 1);

            // 绘制地图
            int plan_map_id = -1;
            if (this->map) {
                // 绘制全局导航地图
                auto id_map = this->map->generate_plan_map();
                plan_map_id = id_map.first;
                plan_map = id_map.second;
                if (this->map_texture == 0) {
                    // 创建地图纹理
                    glGenTextures(1, &map_texture);
                    glBindTexture(GL_TEXTURE_2D, map_texture);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, this->map->size[0], this->map->size[1], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, plan_map.data);
                }
                else {
                    // 仅更新纹理像素
                    glBindTexture(GL_TEXTURE_2D, map_texture);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->map->size[0], this->map->size[1], GL_LUMINANCE, GL_UNSIGNED_BYTE, plan_map.data);
                }
                glEnable(GL_TEXTURE_2D);
                glBegin(GL_QUADS);
                glTexCoord2i(0, 0); glVertex2d(this->map->left_bottom[0], this->map->left_bottom[1]);
                glTexCoord2i(0, 1); glVertex2d(this->map->left_bottom[0], this->map->right_top[1]);
                glTexCoord2i(1, 1); glVertex2d(this->map->right_top[0], this->map->right_top[1]);
                glTexCoord2i(1, 0); glVertex2d(this->map->right_top[0], this->map->left_bottom[1]);
                glEnd();
                glDisable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, 0);
                
                // 绘制任务二家具的局部地图, 需要加锁
                do {
                    std::lock_guard<std::mutex> lock(this->map->mutex_furnitures);
                    // 避免透明物体互相遮挡
                    glDepthMask(false);

                    for (FurnitureMap& fur : this->map->furnitures) {
                        std::string fur_type;
                        cv::Scalar fur_color;
                        switch (fur.check_type()) {
                        case FurnitureMap::FurnitureType::CHAIR:
                            fur_type = "CHAIR";
                            fur_color = cv::Scalar(0, 0x99, 0);
                            break;
                        case FurnitureMap::FurnitureType::DESK:
                            fur_type = "DESK";
                            fur_color = cv::Scalar(0, 0, 0x66);
                            break;
                        case FurnitureMap::FurnitureType::CABILNET:
                            fur_type = "CABILNET";
                            fur_color = cv::Scalar(0x99, 0, 0);
                            break;
                        default:
                            fur_type = "UNKNOWN";
                            fur_color = cv::Scalar();
                        }
                        GLuint texture;
                        try {
                            // 已经存在对应纹理对象
                            texture = this->submap_texture.at(fur.id);
                            glBindTexture(GL_TEXTURE_2D, texture);
                            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fur.size[0], fur.size[1], GL_RGBA, GL_UNSIGNED_BYTE, fur.generate_rgba(fur_color));
                        }
                        catch (const std::out_of_range &) {
                            // 不存在相应纹理对象, 需要创建
                            glGenTextures(1, &texture);
                            this->submap_texture[fur.id] = texture;
                            glBindTexture(GL_TEXTURE_2D, texture);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fur.size[0], fur.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, fur.generate_rgba(fur_color));
                        }
                        glEnable(GL_TEXTURE_2D);
                        glBegin(GL_QUADS);
                        auto corners = fur.get_map_corners();
                        glTexCoord2i(0, 0); glVertex3d(corners[0][0], corners[0][1], 0.01);
                        glTexCoord2i(0, 1); glVertex3d(corners[1][0], corners[1][1], 0.01);
                        glTexCoord2i(1, 1); glVertex3d(corners[2][0], corners[2][1], 0.01);
                        glTexCoord2i(1, 0); glVertex3d(corners[3][0], corners[3][1], 0.01);
                        glEnd();
                        glDisable(GL_TEXTURE_2D);
                        glBindTexture(GL_TEXTURE_2D, 0);
                        // 绘制名称与坐标轴
						try {
							draw_axis(fur.get_universal_pos3d(), 0.5, std::to_string(fur.id) + " " + fur_type);
						}
						catch (const std::exception&) {
							draw_axis(fur.get_pos3d(), 0.5, std::to_string(fur.id) + " " + fur_type);
						}
                    }
                    // 还原深度测试
                    glDepthMask(true);
                } while (0);

                // 绘制任务三障碍物, 需要加锁
                do {
                    std::lock_guard<std::mutex> lock(this->map->mutex_obstacles);
                    int i = 0;
                    for (const Obstacle& obs : Obstacle::get_task3_obstacles(this->map->obstacles)) {
                        // 绘制名称与坐标轴
                        if (obs.check_task3()) {
                            std::ostringstream ss;
                            ss << obs.id << " OBS_" << i + 1;
                            draw_axis(obs.get_pos(), 0.5, ss.str());
                        }
                        i++;
                    }
                } while (0);

                // 绘制任务四煤气罐
                try {
                    auto id = this->map->get_gas_tank_id();
                    auto pos3d = this->map->get_gas_tank_pos();
                    draw_axis(pos3d, 0.5, std::to_string(id) + " GAS_TANK");
                }
                catch (const std::out_of_range&) {
                }
            }

            // 绘制规划信息
            mutex_planner.lock();
            if (planner != NULL) {
                // 绘制轨迹与路径
                planner->mutex_buffer.lock();
                auto car_trajectory = planner->temp_trajectory;;
                planner->mutex_buffer.unlock();
                for (size_t i = 0; i < car_trajectory.size(); i++) {
                    Eigen::Vector2d p1 = car_trajectory[i].pos.translation();
                    Eigen::Vector2d p2 = car_trajectory[i].pos * Eigen::Vector2d(0.1, 0);

                    glBegin(GL_LINES);
                    glColor3f(0, 0, 1);
                    glVertex3d(p1[0], p1[1], 0.01);
                    glColor3f(0, 1, 0);
                    glVertex3d(p2[0], p2[1], 0.05);
                    glEnd();

                    if (i > 0) {
                        Eigen::Vector2d p0 = car_trajectory[i - 1].pos.translation();
                        glColor3f(1, 0, 0);
                        glBegin(GL_LINES);
                        glVertex3d(p0[0], p0[1], 0.01);
                        glVertex3d(p1[0], p1[1], 0.01);
                        glEnd();
                        glBegin(GL_QUADS);
                        glVertex3d(p0[0], p0[1], 0);
                        glVertex3d(p1[0], p1[1], 0);
                        glVertex3d(p1[0], p1[1], car_trajectory[i].vel_linear.norm() * 0.1);
                        glVertex3d(p0[0], p0[1], car_trajectory[i - 1].vel_linear.norm() * 0.1);
                        glEnd();
                    }
                }

                planner->mutex_buffer.lock();
                auto car_path = planner->solution_backend;
                planner->mutex_buffer.unlock();
                for (size_t i = 0; i < car_path.size(); i++) {
                    Eigen::Vector2d p1 = car_path[i].translation();
                    Eigen::Vector2d p2 = car_path[i] * Eigen::Vector2d(0.1, 0);

                    //glBegin(GL_LINES);
                    //glColor3f(0, 0, 1);
                    //glVertex3d(p1[0], p1[1], 0.01);
                    //glColor3f(0, 1, 0);
                    //glVertex3d(p2[0], p2[1], 0.05);
                    //glEnd();
                    glDepthMask(false);
                    glColor3f(1, 0, 0);
                    std::ostringstream ss;
                    ss << i << " [" << p1[0] << ", " << p1[1] << "]";
                    pangolin::GlFont::I().Text(ss.str().c_str()).Draw(p1[0], p1[1], 0.01);
                    glDepthMask(true);

                    if (i > 0) {
                        Eigen::Vector2d p0 = car_path[i - 1].translation();
                        glColor3f(1, 0, 0);
                        glBegin(GL_LINES);
                        glVertex3d(p0[0], p0[1], 0.01);
                        glVertex3d(p1[0], p1[1], 0.01);
                        glEnd();
                    }
                }

                // 绘制碰撞点
                planner->mutex_buffer.lock();
                auto cls_point = planner->temp_cls_point;
                planner->mutex_buffer.unlock();

                glColor3f(1, 0, 0);
                glBegin(GL_LINES);
                glVertex3d(cls_point[0], cls_point[1], 0);
                glVertex3d(cls_point[0], cls_point[1], 0.2);
                glEnd();
            }
            mutex_planner.unlock();

            // draw car
            if (this->car_size[0] > 0 && this->car_size[1] > 0) {
                auto LF = this->car_pos * Eigen::Vector2d(this->car_size[0] / 2, this->car_size[1] / 2);
                auto LB = this->car_pos * Eigen::Vector2d(-this->car_size[0] / 2, this->car_size[1] / 2);
                auto RB = this->car_pos * Eigen::Vector2d(-this->car_size[0] / 2, -this->car_size[1] / 2);
                auto RF = this->car_pos * Eigen::Vector2d(this->car_size[0] / 2, -this->car_size[1] / 2);

                // top
                //glBegin(GL_QUADS);
                //glColor3d(1, 0.6, 0);
                //glVertex3d(RF[0], RF[1], 0.1);
                //glVertex3d(LF[0], LF[1], 0.1);
                //glColor3d(1, 0, 0);
                //glVertex3d(LB[0], LB[1], 0.1);
                //glVertex3d(RB[0], RB[1], 0.1);
                //glEnd();

                // front
                glBegin(GL_QUADS);
                glColor3d(1, 0.6, 0);
                glVertex3d(RF[0], RF[1], 0.1);
                glVertex3d(LF[0], LF[1], 0.1);
                glColor3d(1, 1, 0);
                glVertex3d(LF[0], LF[1], 0);
                glVertex3d(RF[0], RF[1], 0);
                glEnd();

                // left
                glBegin(GL_QUADS);
                glColor3d(1, 0.6, 0);
                glVertex3d(LF[0], LF[1], 0.1);
                glVertex3d(LF[0], LF[1], 0);
                glVertex3d(LB[0], LB[1], 0);
                glVertex3d(LB[0], LB[1], 0.1);
                glEnd();

                // back
                glBegin(GL_QUADS);
                glColor3d(1, 0.6, 0);
                glVertex3d(LB[0], LB[1], 0.1);
                glVertex3d(LB[0], LB[1], 0);
                glVertex3d(RB[0], RB[1], 0);
                glVertex3d(RB[0], RB[1], 0.1);
                glEnd();

                // left
                glBegin(GL_QUADS);
                glColor3d(1, 0.6, 0);
                glVertex3d(RF[0], RF[1], 0.1);
                glVertex3d(RF[0], RF[1], 0);
                glVertex3d(RB[0], RB[1], 0);
                glVertex3d(RB[0], RB[1], 0.1);
                glEnd();
            }

            // 文本显示
            GLfloat x0 = 10;
            GLfloat y0 = 10;
            GLfloat dy = 15;
            glColor3f(1.0f, 0.1f, 0.1f);

            // 显示机器人状态
            mutex_data.lock();
            auto robot_status_lines = status_log;
            mutex_data.unlock();
            for (auto p = robot_status_lines.rbegin(); p != robot_status_lines.rend(); p++, y0 += dy) {
                pangolin::GlFont::I().Text(*p).DrawWindow(x0, y0);
            }
            // 行分隔
            pangolin::GlFont::I().Text("----------" + std::to_string(plan_map_id) + "----------").DrawWindow(x0, y0);
            y0 += dy;
            // 显示调试日志
            mutex_data.lock();
            for (
                int i = (debug_log_index + debug_log.size() - 1) % debug_log.size();
                i != debug_log_index;
                i = (i + debug_log.size() - 1) % debug_log.size(), y0 += dy
            ) {
                if (!debug_log[i].empty()) {
                    pangolin::GlFont::I().Text(debug_log[i]).DrawWindow(x0, y0);
                }
            }
            mutex_data.unlock();

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }
    }
};