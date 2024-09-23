#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"
#include "preprocess.h"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

// 比较两个点云点的曲率大小
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
// IMU处理和去畸变
class ImuProcess
{
 public:
  //内存对齐
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  //重置IMU处理器
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  //设置外参
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  //处理IMU数据
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  //处理文件流
  ofstream fout_imu;
  
  //设置预处理的参数
  V3D cov_acc;//加速度协方差
  V3D cov_gyr;//
  V3D cov_acc_scale;    //加速度噪声尺度
  V3D cov_gyr_scale;    //角速度噪声尺度
  V3D cov_bias_gyr;    //
  V3D cov_bias_acc;
  double first_lidar_time;    //激光雷达的第一个时间戳
  int lidar_type;        //激光雷达的类型

 private:
  //初始化IMU处理器
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  //去畸变点云
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);
  //当前点云
  PointCloudXYZI::Ptr cur_pcl_un_;
  //最后一个IMU数据
  sensor_msgs::ImuConstPtr last_imu_;
  //IMU数据队列
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  //每个IMU测量时刻IMU的姿态
  vector<Pose6D> IMUpose;
  //存储每个IMU测量时刻的激光雷达相对于IMU的姿态。M3D 是一个3x3的旋转矩阵
  vector<M3D>    v_rot_pcl_;
  //激光雷达相对于IMU的旋转矩阵（存疑）
  M3D Lidar_R_wrt_IMU;
  //激光雷达相对于IMU的平移向量（存疑）
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;    //激光雷达原点处的平均加速度
  V3D mean_gyr;    //激光雷达原点处的平均角速度
  
  V3D angvel_last;    //上一个IMU测量时刻的角速度
  V3D acc_s_last;    //上一个IMU测量时刻的加速度
  //一开始的时间戳
  double start_timestamp_;
  //最后时刻的雷达时间戳
  double last_lidar_end_time_;
  
  int    init_iter_num = 1;    //迭代次数
  bool   b_first_frame_ = true;     //true：是第一个激光雷达帧
  bool   imu_need_init_ = true;    //true：初始化IMU
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  // 初始化迭代次数
  init_iter_num = 1;
  // 初始化过程噪声协方差矩阵 Q
  Q = process_noise_cov();
  // 初始化加速度协方差矩阵 cov_acc
  cov_acc       = V3D(0.1, 0.1, 0.1);
  // 初始化角速度协方差矩阵 cov_gyr
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  // 初始化陀螺仪偏差协方差矩阵 cov_bias_gyr
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  // 初始化加速度偏差协方差矩阵 cov_bias_acc
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  // 初始化激光雷达原点处的平均加速度 mean_acc
  mean_acc      = V3D(0, 0, -1.0);
  // 初始化激光雷达原点处的平均角速度 mean_gyr
  mean_gyr      = V3D(0, 0, 0);
  // 初始化上一个IMU测量时刻的角速度 angvel_last
  angvel_last     = Zero3d;
  // 初始化激光雷达相对于IMU的平移向量 Lidar_T_wrt_IMU
  Lidar_T_wrt_IMU = Zero3d;
  // 初始化激光雷达相对于IMU的旋转矩阵 Lidar_R_wrt_IMU
  Lidar_R_wrt_IMU = Eye3d;
  // 初始化最后一个IMU数据 last_imu_
  last_imu_.reset(new sensor_msgs::Imu());
}


ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  // 重置激光雷达原点处的平均加速度 mean_acc
  mean_acc      = V3D(0, 0, -1.0);
  // 重置激光雷达原点处的平均角速度 mean_gyr
  mean_gyr      = V3D(0, 0, 0);
  // 重置上一个IMU测量时刻的角速度 angvel_last
  angvel_last       = Zero3d;
  // 重置是否需要初始化IMU的标志 imu_need_init_
  imu_need_init_    = true;
  // 重置第一个激光雷达时间戳 start_timestamp_
  start_timestamp_  = -1;
  // 重置初始化迭代次数 init_iter_num
  init_iter_num     = 1;
  // 清除IMU数据队列 v_imu_
  v_imu_.clear();
  // 清除IMU姿态队列 IMUpose
  IMUpose.clear();
  // 重置最后一个IMU数据 last_imu_
  last_imu_.reset(new sensor_msgs::Imu());
  // 重置当前未畸变的点云数据 cur_pcl_un_
  cur_pcl_un_.reset(new PointCloudXYZI());
}


void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  // 获取IMU相对于激光雷达的平移向量 Lidar_T_wrt_IMU
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  // 获取IMU相对于激光雷达的旋转矩阵 Lidar_R_wrt_IMU
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}


void ImuProcess::set_extrinsic(const V3D &transl)
{
  // 设置激光雷达相对于IMU的平移向量 Lidar_T_wrt_IMU
  Lidar_T_wrt_IMU = transl;
  // 设置激光雷达相对于IMU的旋转矩阵 Lidar_R_wrt_IMU 为单位矩阵
  Lidar_R_wrt_IMU.setIdentity();
}


void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  // 设置激光雷达相对于IMU的平移向量 Lidar_T_wrt_IMU
  Lidar_T_wrt_IMU = transl;
  // 设置激光雷达相对于IMU的旋转矩阵 Lidar_R_wrt_IMU
  Lidar_R_wrt_IMU = rot;
}


void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  //定义当前加速度和角速度
  V3D cur_acc, cur_gyr;
  
  //如果这是第一个激光雷达帧，则重置所有状态
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    // 计算平均加速度和角速度
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }
    //遍历所有IMU测量数据
  for (const auto &imu : meas.imu)
  {
     // 获取当前IMU的加速度和角速度
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
 
    // 更新平均加速度和角速度
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
    //更新 N 的值
    N ++;
  }
  // 获取当前状态估计
  state_ikfom init_state = kf_state.get_x();
  // 初始化重力向量
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  //初始化陀螺仪偏差向量
  init_state.bg  = mean_gyr;
  // 设置激光雷达相对于IMU的平移和旋转
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
    // 更新状态估计
  kf_state.change_x(init_state);

  //获取当前协方差矩阵
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  //初始化协方差矩阵
  init_P.setIdentity();
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  init_P(21,21) = init_P(22,22) = 0.00001; 
  //更新协方差矩阵
  kf_state.change_P(init_P);
  //更新IMU最新数据
  last_imu_ = meas.imu.back();

}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();

  double pcl_beg_time = meas.lidar_beg_time;
  double pcl_end_time = meas.lidar_end_time;

    if (lidar_type == MARSIM) {
        pcl_beg_time = last_lidar_end_time_;
        pcl_end_time = meas.lidar_beg_time;
    }

    /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;

  if(lidar_type != MARSIM){
      auto it_pcl = pcl_out.points.end() - 1;
      for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
      {
          auto head = it_kp - 1;
          auto tail = it_kp;
          R_imu<<MAT_FROM_ARRAY(head->rot);
          // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
          vel_imu<<VEC_FROM_ARRAY(head->vel);
          pos_imu<<VEC_FROM_ARRAY(head->pos);
          acc_imu<<VEC_FROM_ARRAY(tail->acc);
          angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

          for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
          {
              dt = it_pcl->curvature / double(1000) - head->offset_time;

              /* Transform to the 'end' frame, using only the rotation
               * Note: Compensation direction is INVERSE of Frame's moving direction
               * So if we want to compensate a point at timestamp-i to the frame-e
               * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
              M3D R_i(R_imu * Exp(angvel_avr, dt));

              V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
              V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
              V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!

              // save Undistorted points and their rotation
              it_pcl->x = P_compensate(0);
              it_pcl->y = P_compensate(1);
              it_pcl->z = P_compensate(2);

              if (it_pcl == pcl_out.points.begin()) break;
          }
      }
  }
}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}