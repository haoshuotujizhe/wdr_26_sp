#include "planner.hpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Planner::Planner(const std::string & config_path,const int ifdouble)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_doubleyaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan(*target, bullet_speed);
}

Plan_double Planner::plan_double(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  return plan_double(*target, bullet_speed);
}

Plan_double Planner::plan_double(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(3);
  x0 << traj(0, 0), traj(0, 0)*0.5, traj(0, 0)*0.5;
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = Eigen::MatrixXd::Zero(3,HORIZON);
  yaw_solver_->work->Xref.block(0,0,1, HORIZON) = traj.block(0, 0, 1, HORIZON);
  // yaw_solver_->work->Xref.block(0,0,1, HORIZON) = traj.block(0, 0, 1, HORIZON)*0.7;
  // yaw_solver_->work->Xref.block(0,0,1, HORIZON) = traj.block(0, 0, 1, HORIZON)*0.3;
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  Eigen::VectorXd x01(2);
  x01 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x01);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan_double plan_double;
  plan_double.control = true;

  plan_double.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan_double.target_pitch = traj(2, HALF_HORIZON);

  plan_double.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan_double.yaw_vel = yaw_solver_->work->u(0, HALF_HORIZON)+yaw_solver_->work->u(1, HALF_HORIZON);
  // plan_double.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);
  plan_double.yaw_acc = yaw_solver_->work->u(2, HALF_HORIZON)+yaw_solver_->work->u(3, HALF_HORIZON);

  plan_double.yaw_big = tools::limit_rad(yaw_solver_->work->x(1, HALF_HORIZON) + yaw0);
  plan_double.yaw_vel_big = yaw_solver_->work->u(0, HALF_HORIZON);
  plan_double.yaw_acc_big = yaw_solver_->work->u(2, HALF_HORIZON);

  plan_double.yaw_small = tools::limit_rad(yaw_solver_->work->x(2, HALF_HORIZON) + yaw0);
  plan_double.yaw_vel_small = yaw_solver_->work->u(1, HALF_HORIZON);
  plan_double.yaw_acc_small = yaw_solver_->work->u(3, HALF_HORIZON);

  plan_double.yaw_sum = plan_double.yaw_big+plan_double.yaw_small;

  plan_double.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan_double.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan_double.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan_double.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan_double;
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_doubleyaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_big_yaw_acc = tools::read<double>(yaml, "max_big_yaw_acc");
  auto max_small_yaw_acc = tools::read<double>(yaml, "max_small_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");
  //clang-format off
  // Eigen::MatrixXd A{{1, 0, 0, DT, DT}, \
  //                   {0, 1, 0, DT, 0 }, \
  //                   {0, 0, 1, 0 , DT}, \
  //                   {0, 0, 0, 1 , 0 }, \
  //                   {0, 0, 0, 0 , 1 }  };
  Eigen::MatrixXd A{{0, 1, 1}, \
                    {0, 1, 0}, \
                    {0, 0, 1}  };
  
  Eigen::MatrixXd B{{ 0, 0,  0,  0}, \
                    { 1, 0, DT,  0}, \
                    { 0, 1,  0, DT}};
  //clang-format on
  Eigen::VectorXd f{{0, 0, 0}};
  Eigen::Matrix<double, 3, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 4, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 3, 4, HORIZON, 0);

  Eigen::MatrixXd x_max(3, HORIZON);
  x_max.row(0).setConstant(1e17); // No limit on angle
  x_max.row(1).setConstant(1e17);  // No limit on angle1
  x_max.row(2).setConstant(M_PI/6);  // limit on angle2

  Eigen::MatrixXd x_min(3, HORIZON);
  x_min.row(0).setConstant(-1e17); // No limit on angle
  x_min.row(1).setConstant(-1e17); // Min velocity 1
  x_min.row(2).setConstant(-M_PI/6); // Min velocity 2

  Eigen::MatrixXd u_max(4, HORIZON - 1);
  u_max.row(0).setConstant(0.001); // Max velocity 1  
  u_max.row(1).setConstant(1.0);  // Max velocity 2
  u_max.row(2).setConstant(max_big_yaw_acc);  // Limit for u(0)
  u_max.row(3).setConstant(max_small_yaw_acc); // Limit for u(1)

  Eigen::MatrixXd u_min(4, HORIZON - 1);
  u_min.row(0).setConstant(-0.001); // Max velocity 1  
  u_min.row(1).setConstant(-1.0);  // Max velocity 2
  u_min.row(2).setConstant(-max_big_yaw_acc); // Limit for u(0)
  u_min.row(3).setConstant(-max_small_yaw_acc); // Limit for u(1)
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  // 添加线性等式约束：yaw - yaw_big - yaw_small = 0
  // 转换为两个不等式：
  // yaw - yaw_big - yaw_small <= 0  和  yaw - yaw_big - yaw_small >= 0
  
  Eigen::MatrixXd Alin_x(2, 3);  // 2个约束，3个状态变量
  Alin_x << 1, -1, -1,   // yaw - yaw_big - yaw_small <= 0
           -1,  1,  1;   // -yaw + yaw_big + yaw_small <= 0
  
  Eigen::VectorXd blin_x(2);
  blin_x << 0, 0;  // 右侧都是 0
  
  // 没有控制输入的线性约束
  Eigen::MatrixXd Alin_u(0, 2);  // 0个约束
  Eigen::VectorXd blin_u(0);
  
  tiny_set_linear_constraints(yaw_solver_, Alin_x, blin_x, Alin_u, blin_u);

  // 启用线性约束
  yaw_solver_->settings->en_state_linear = 1;
  yaw_solver_->settings->en_input_linear = 0;

  yaw_solver_->settings->max_iter = 10; 
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim