


#include "custom_diff_controller/odometry.hpp"

namespace custom_diff_controller{

    Odometry::Odometry(size_t velocity_rolling_window_size)
    : timestamp_(0.0),
    x_(0.0),
    y_(0.0),
    heading_(0.0),
    linear_(0.0),
    angular_(0.0),
    wheel_seperation_(0.0),
    left_wheel_radius_(0.0),
    right_wheel_radius_(0.0),
    left_wheel_old_pos_(0.0),
    right_wheel_old_pos_(0.0),
    velocity_rolling_window_size_(velocity_rolling_window_size),
    linear_accumulator_(velocity_rolling_window_size),
    angular_accumulator_(velocity_rolling_window_size)
    {
    }

    void Odometry::init(const rclcpp::Time & time){
        resetAccumulators();
        timestamp_ = time;
    }

    bool Odometry::update(double left_pos, double right_pos, const rclcpp::Time & time){
        const double dt = time.seconds() - timestamp_.seconds();
        if(dt < 0.0001){
            return false; //interval too small to  integrate with
        }
        // get the current wheel joint position
        const double left_wheel_curr_pos = left_pos*left_wheel_radius_;
        const double right_wheel_curr_pos = right_pos*right_wheel_radius_;
        //  Estimate velocity of  wheels using old and current position
        const double left_wheel_est_vel  = left_wheel_curr_pos - left_wheel_old_pos_;
        const double right_wheel_est_vel = right_wheel_curr_pos - right_wheel_old_pos_;
        // update the old with the current pos
        left_wheel_old_pos_ =  left_wheel_curr_pos;
        right_wheel_old_pos_ = right_wheel_curr_pos;
        updateFromVelocity(left_wheel_est_vel,right_wheel_est_vel,time);
        return true;
    }
    bool Odometry::updateFromVelocity(double left_vel, double right_vel, const rclcpp::Time & time)
    {
    const double dt = time.seconds() - timestamp_.seconds();
    if (dt < 0.0001)
    {
        return false;  // Interval too small to integrate with
    }
    // Compute linear and angular diff:
    const double linear = (left_vel + right_vel) * 0.5;
    // Now there is a bug about scout angular velocity
    const double angular = (right_vel - left_vel) / wheel_seperation_;

    // Integrate odometry:
    integrateExact(linear, angular);

    timestamp_ = time;

    // Estimate speeds using a rolling mean to filter them out:
    linear_accumulator_.accumulate(linear / dt);
    angular_accumulator_.accumulate(angular / dt);

    linear_ = linear_accumulator_.getRollingMean();
    angular_ = angular_accumulator_.getRollingMean();

    return true;
    }
    
    
    void Odometry::integrateRungeKutta2(double linear, double angular){

        const double direction = heading_ + angular*0.5;
        x_ += linear*std::cos(direction);
        y_ += linear*std::sin(direction);
        heading_ +=angular;
    }
    
    
    void Odometry::integrateExact(double linear, double angular){
        if (fabs(angular) < 1e-6 )
        {
            integrateRungeKutta2(linear,angular);

        }else{
            const double heading_old = heading_;
            const double r = linear/angular;
            heading_ +=angular;
            x_+=   r*(std::sin(heading_) - std::sin(heading_old));
            y_+= - r*(std::cos(heading_) - std::cos(heading_old));
        }
    }    




    // confused
    void Odometry::updateOpenLoop(double linear, double angular, const rclcpp::Time & time){
        // save last linear and angular velocity:
        linear_ = linear;
        angular_ = angular;

        // integrate odometry
        const double dt = time.seconds()  - timestamp_.seconds();
        timestamp_  = time;
        integrateExact(linear*dt,angular*dt);
    }


    void Odometry::resetOdometry()
    {
        x_ = 0.0;
        y_ = 0.0;
        heading_ = 0.0;
    }

    void Odometry::setWheelParams(
        double wheel_seperation, double left_wheel_radius, double right_wheel_radius
    ){

        wheel_seperation_ = wheel_seperation;
        left_wheel_radius_ = left_wheel_radius;
        right_wheel_radius_ = right_wheel_radius;
    }

    void Odometry::setVelocityRollingWindowSize(size_t velocity_rollwing_window_size){

            velocity_rolling_window_size_ = velocity_rollwing_window_size;
            resetAccumulators();

    }




    void Odometry::resetAccumulators()
    {
    linear_accumulator_ = RollingMeanAccumulator(velocity_rolling_window_size_);
    angular_accumulator_ = RollingMeanAccumulator(velocity_rolling_window_size_);
    }

}