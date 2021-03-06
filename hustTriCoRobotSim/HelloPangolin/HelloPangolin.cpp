// HelloPangolin.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "../hustTriCoRobotSim/gui.h"
#include "../hustTriCoRobotSim/map.h"

static DebugGUI DebugWindow;
static GlobalMap RobotMapper(Eigen::Vector2d(-14, -15), Eigen::Vector2d(4, 4), 0.02);

int main(int /*argc*/, char** /*argv*/)
{
    DebugWindow.set_map(&RobotMapper);
    DebugWindow.set_car_size(Eigen::Vector2d(0.8, 0.6));
    DebugWindow.set_car_position(Eigen::Transform<double, 2, Eigen::Affine>(Eigen::Translation2d(-3.3, 1.722)));

    RobotMapper.update(false, Eigen::Affine2d(Eigen::Translation2d(0, 0)), Eigen::Vector2d(1, 1));

    DebugWindow.run();
	return 0;
}