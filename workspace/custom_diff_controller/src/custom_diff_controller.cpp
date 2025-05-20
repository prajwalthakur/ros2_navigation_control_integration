

#include "custom_diff_controller/custom_diff_controller.hpp"
namespace
{
    constexpr auto DEFAULT_COMMAND_TOPIC = "~/cmd_vel";
    constexpr auto DEFAULT_COMMAND_UNSTAMPED_TOPIC = "~/cmd_vel_unstamped";
    constexpr auto DEFAULT_COMMAND_OUT_TOPIC = "~/cmd_vel_out";
    constexpr auto DEFAULT_ODOMETRY_TOPIC = "~/odom";
    constexpr auto DEFAULT_TRANSFORM_TOPIC = "/tf";
}  // namespace

namespace custom_diff_controller
{
    using namespace std::chrono_literals;
    using controller_interface::interface_configuration_type;
    using controller_interface::InterfaceConfiguration;
    using hardware_interface::HW_IF_POSITION;
    using hardware_interface::HW_IF_VELOCITY;
    using lifecycle_msgs::msg::State;

CustomDiffController::CustomDiffController():controller_interface::ControllerInterface() {

}

const char* CustomDiffController::feedback_type()const{
    return params_.position_feedback?HW_IF_POSITION:HW_IF_VELOCITY;
}

controller_interface::CallbackReturn CustomDiffController::on_init(){

    try{

        // create parameter listener and get the parameters
        param_listener_ = std::make_shared<ParamListener>(get_node());
        params_ = param_listener_->get_params();

    }
    catch(const std::exception & e){

        fprintf(stderr, "Exception thrown during init stage with message: %s \n",e.what());
        return controller_interface::CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
}

InterfaceConfiguration CustomDiffController::command_interface_configuration() const {

    std::vector<std::string> conf_names;
    for(const auto &joint_name: params_.left_wheel_names){
        conf_names.push_back(joint_name + "/"+ HW_IF_VELOCITY);
    }
    for(const auto& joint_name: params_.right_wheel_names){
        conf_names.push_back(joint_name + "/"+ HW_IF_VELOCITY);
    }
    
    return {interface_configuration_type::INDIVIDUAL, conf_names};
}

InterfaceConfiguration CustomDiffController::state_interface_configuration() const{

    std::vector<std::string> conf_names;
    for(const auto &joint_name: params_.left_wheel_names){
        conf_names.push_back(joint_name + "/"+ feedback_type()); //get the pose
    }
    for(const auto& joint_name: params_.right_wheel_names){
        conf_names.push_back(joint_name + "/"+ feedback_type()); //get the pose
    }
    
    return {interface_configuration_type::INDIVIDUAL, conf_names};
}
void CustomDiffController::halt(){

    const auto halt_wheels =[](auto & wheel_handles){
        for(const auto& wheel_handle: wheel_handles){
            wheel_handle.velocity.get().set_value(0.0);

        }
    };
    halt_wheels(registered_left_wheel_handles_);
    halt_wheels(registered_right_wheel_handles_);
}


controller_interface::return_type CustomDiffController::update(const rclcpp::Time & time, const rclcpp::Duration & period){

    auto logger = get_node()->get_logger() ;//reference to the node pointer's logger
    if(get_state().id()==State::PRIMARY_STATE_INACTIVE){
        //# This state represents a node that is not currently performing any processing.
        // uint8 PRIMARY_STATE_INACTIVE = 2
        if(!is_halted){
            halt();
            is_halted = true;
        }
        return controller_interface::return_type::OK;
    }
    // std::shared_ptr<Twist> last_command_msg;
    // received_velocity_msg_ptr_.get(last_command_msg);
    std::shared_ptr<Twist> last_command_msg = *(received_velocity_msg_ptr_.readFromRT());
    if(last_command_msg==nullptr){
        RCLCPP_WARN(logger,"velocity message was a nullptr");
        return controller_interface::return_type::ERROR;
    }

    const auto duration_since_last_command = time - last_command_msg->header.stamp;
    if(duration_since_last_command > cmd_vel_timeout_){
        last_command_msg->twist.linear.x = 0.0;
        last_command_msg->twist.angular.z = 0.0;
        RCLCPP_WARN_ONCE(logger,"command timed-out, message will only print once ");
    }else if ( ! (std::isfinite(last_command_msg->twist.linear.x) && std::isfinite(last_command_msg->twist.angular.z))) {

        RCLCPP_WARN_SKIPFIRST_THROTTLE(logger, *get_node()->get_clock(), cmd_vel_timeout_.seconds() * 1000,
            "Command message contains NaNs. Not updating reference interfaces.");
        return controller_interface::return_type::OK;
    }
    // command may be limited further by speedlimit
    Twist command = *last_command_msg;
    // command.twist.linear.x = 0.5;
    // command.twist.angular.z = 0.1;
    double & linear_command = command.twist.linear.x;
    double & angular_command = command.twist.angular.z;
    previous_update_timestamp_ = time;
    // Apply (possibly new) multipliers:
    const double wheel_separation = params_.wheel_separation_multiplier * params_.wheel_separation;
    const double left_wheel_radius = params_.left_wheel_radius_multiplier * params_.wheel_radius;
    const double right_wheel_radius = params_.right_wheel_radius_multiplier * params_.wheel_radius;
    //RCLCPP_INFO(logger,"before params_.open_loop");
    //If set to true the odometry of the robot will be calculated from the commanded values and not from feedback
    if(params_.open_loop){
        odometry_.updateOpenLoop(linear_command,angular_command,time);
    }else{

        double left_feedback_mean = 0.0;
        double right_feedback_mean = 0.0;
        for(size_t index=0;index < static_cast<size_t>(wheels_per_side_);++index)
        {
            const double left_feedback = registered_left_wheel_handles_[index].feedback.get().get_value();
            const double right_feedback = registered_right_wheel_handles_[index].feedback.get().get_value();
            if (std::isnan(left_feedback) || std::isnan(right_feedback))
            {
                RCLCPP_ERROR(
                logger, "Either the left or right wheel %s is invalid for index [%zu]", feedback_type(),
                index);
                return controller_interface::return_type::ERROR;
            }

            left_feedback_mean += left_feedback;
            right_feedback_mean += right_feedback;
        }
        left_feedback_mean/=static_cast<double>(wheels_per_side_);
        right_feedback_mean/=static_cast<double>(wheels_per_side_);

        if (params_.position_feedback)
        {
        odometry_.update(left_feedback_mean, right_feedback_mean, time);
        }
        else
        {
        odometry_.updateFromVelocity(
            left_feedback_mean * left_wheel_radius * period.seconds(),
            right_feedback_mean * right_wheel_radius * period.seconds(), time);
        }
    }
    tf2::Quaternion Orientation;
    Orientation.setRPY(0.0,0.0, odometry_.getHeading());
    bool should_publish = false;
    try
    {
        if (previous_publish_timestamp_ + publish_period_ < time)
        {
        previous_publish_timestamp_ += publish_period_;
        should_publish = true;
        }
    }
    catch (const std::runtime_error &)
    {
        // Handle exceptions when the time source changes and initialize publish timestamp
        previous_publish_timestamp_ = time;
        should_publish = true;
    }    
    if (should_publish)
    {
        if (realtime_odometry_publisher_->trylock())
        {
        auto & odometry_message = realtime_odometry_publisher_->msg_;
        odometry_message.header.stamp = time;
        odometry_message.pose.pose.position.x = odometry_.getX();
        odometry_message.pose.pose.position.y = odometry_.getY();
        odometry_message.pose.pose.orientation.x = Orientation.x();
        odometry_message.pose.pose.orientation.y = Orientation.y();
        odometry_message.pose.pose.orientation.z = Orientation.z();
        odometry_message.pose.pose.orientation.w = Orientation.w();
        odometry_message.twist.twist.linear.x = odometry_.getLinear();
        odometry_message.twist.twist.angular.z = odometry_.getAngular();
        realtime_odometry_publisher_->unlockAndPublish();
        }

        if (params_.enable_odom_tf && realtime_odometry_transform_publisher_->trylock())
        {
            auto & transform = realtime_odometry_transform_publisher_->msg_.transforms.front();
            transform.header.stamp = time;
            transform.transform.translation.x = odometry_.getX();
            transform.transform.translation.y = odometry_.getY();
            transform.transform.rotation.x = Orientation.x();
            transform.transform.rotation.y = Orientation.y();
            transform.transform.rotation.z = Orientation.z();
            transform.transform.rotation.w = Orientation.w();
            realtime_odometry_transform_publisher_->unlockAndPublish();
        }   
    }
    auto & last_command = previous_commands_.back().twist;
    auto & second_to_last_command = previous_commands_.front().twist;

        // RCLCPP_INFO(
        // logger,
        // "before limiteer  linear = %.3f  rad/s | angular = %.3f  rad/s",
        // linear_command,
        // angular_command);
    limiter_linear_.limit(linear_command, last_command.linear.x, second_to_last_command.linear.x, period.seconds());
    limiter_angular_.limit(angular_command, last_command.angular.z, second_to_last_command.angular.z, period.seconds());
    previous_commands_.pop();
    previous_commands_.emplace(command);
    // RCLCPP_INFO(logger," after limiter");
    //    Publish limited velocity
    if (publish_limited_velocity_ && realtime_limited_velocity_publisher_->trylock())
    {
        auto & limited_velocity_command = realtime_limited_velocity_publisher_->msg_;
        limited_velocity_command.header.stamp = time;
        limited_velocity_command.twist = command.twist;
        realtime_limited_velocity_publisher_->unlockAndPublish();
    }
    // RCLCPP_INFO(logger," before angular_command");
    // Compute wheels velocities:
    const double velocity_left =
    (linear_command - angular_command * wheel_separation / 2.0) / left_wheel_radius;
    const double velocity_right =
    (linear_command + angular_command * wheel_separation / 2.0) / right_wheel_radius;

    // Set wheels velocities:
    for (size_t index = 0; index < static_cast<size_t>(wheels_per_side_); ++index)
    {
        // RCLCPP_INFO(
        // logger,
        // "inside cmd wheel velocities  left = %.3f  rad/s | right = %.3f  rad/s",
        // velocity_left,
        // velocity_right);
    registered_left_wheel_handles_[index].velocity.get().set_value(velocity_left);
    registered_right_wheel_handles_[index].velocity.get().set_value(velocity_right);
    }
    // RCLCPP_INFO(
    // logger,
    // "outside cmd wheel velocities  left = %.3f  rad/s | right = %.3f  rad/s",
    // velocity_left,
    // velocity_right);
    // RCLCPP_INFO(logger,"update before return");
    return controller_interface::return_type::OK;    
}
controller_interface::CallbackReturn CustomDiffController::on_configure(const rclcpp_lifecycle::State &){
    auto logger = get_node()->get_logger();

    // update parameters if they have changed
    if (param_listener_->is_old(params_))
    {
    params_ = param_listener_->get_params();
    RCLCPP_INFO(logger, "Parameters were updated");
    }

    if (params_.left_wheel_names.size() != params_.right_wheel_names.size())
    {
    RCLCPP_ERROR(
        logger, "The number of left wheels [%zu] and the number of right wheels [%zu] are different",
        params_.left_wheel_names.size(), params_.right_wheel_names.size());
    return controller_interface::CallbackReturn::ERROR;
    }

    if (params_.left_wheel_names.empty())
    {
    RCLCPP_ERROR(logger, "Wheel names parameters are empty!");
    return controller_interface::CallbackReturn::ERROR;
    }

    const double wheel_separation = params_.wheel_separation_multiplier * params_.wheel_separation;
    const double left_wheel_radius = params_.left_wheel_radius_multiplier * params_.wheel_radius;
    const double right_wheel_radius = params_.right_wheel_radius_multiplier * params_.wheel_radius;

    odometry_.setWheelParams(wheel_separation, left_wheel_radius, right_wheel_radius);
    odometry_.setVelocityRollingWindowSize(static_cast<size_t>(params_.velocity_rolling_window_size));

    //cmd_vel_timeout_ = std::chrono::milliseconds{static_cast<int>(params_.cmd_vel_timeout * 1000.0)};
    cmd_vel_timeout_ = rclcpp::Duration::from_seconds(params_.cmd_vel_timeout);
    publish_limited_velocity_ = params_.publish_limited_velocity;
    use_stamped_vel_ = params_.use_stamped_vel;

    limiter_linear_ = SpeedLimiter(
    params_.linear.x.has_velocity_limits, params_.linear.x.has_acceleration_limits,
    params_.linear.x.has_jerk_limits, params_.linear.x.min_velocity, params_.linear.x.max_velocity,
    params_.linear.x.min_acceleration, params_.linear.x.max_acceleration, params_.linear.x.min_jerk,
    params_.linear.x.max_jerk);

    limiter_angular_ = SpeedLimiter(
    params_.angular.z.has_velocity_limits, params_.angular.z.has_acceleration_limits,
    params_.angular.z.has_jerk_limits, params_.angular.z.min_velocity,
    params_.angular.z.max_velocity, params_.angular.z.min_acceleration,
    params_.angular.z.max_acceleration, params_.angular.z.min_jerk, params_.angular.z.max_jerk);

    if(!reset()){
        return controller_interface::CallbackReturn::ERROR;
    }
    // left and right sides are both equal at this point
    wheels_per_side_ = static_cast<int>(params_.left_wheel_names.size());

    if (publish_limited_velocity_)
    {
    limited_velocity_publisher_ =
        get_node()->create_publisher<Twist>(DEFAULT_COMMAND_OUT_TOPIC, rclcpp::SystemDefaultsQoS());
    realtime_limited_velocity_publisher_ =
        std::make_shared<realtime_tools::RealtimePublisher<Twist>>(limited_velocity_publisher_);
    }
    //const Twist empty_twist;
    // increate the counter to 1 for temporary pointer, 
    //then copy it to the thing_ then temp_ptr gets destroyed but we still we have a reference to the pointer through thing
    // received_velocity_msg_ptr_.set(std::make_shared<Twist>(empty_twist));    
    // // Fill last two commands with default constructed commands
    // previous_commands_.emplace(empty_twist);
    // previous_commands_.emplace(empty_twist);
    if(use_stamped_vel_){
        velocity_command_subscriber_ = get_node()->create_subscription<Twist>(
            DEFAULT_COMMAND_TOPIC, rclcpp::SystemDefaultsQoS(),
            [this](const std::shared_ptr<Twist> msg)->void {
                RCLCPP_INFO(
                get_node()->get_logger(),
                "in stampedddr = %.3f  rad/s | angular = %.3f  rad/s",
                msg->twist.linear.x,
                msg->twist.angular.z);

                if (!subscriber_is_active_){
                    RCLCPP_WARN(get_node()->get_logger(),"CANT ACCEPT new commands, subscriber is not active");
                    return;
                }else{
                    if( (msg->header.stamp.sec==0) && (msg->header.stamp.nanosec==0) ){
                        RCLCPP_WARN(get_node()->get_logger(),"CANT ACCEPT msg recieved has 0 value in timestamp");
                        return;
                    }
                }
                received_velocity_msg_ptr_.writeFromNonRT(msg);  
            });
    }else{
    velocity_command_unstamped_subscriber_ =
      get_node()->create_subscription<geometry_msgs::msg::Twist>(
         DEFAULT_COMMAND_UNSTAMPED_TOPIC, rclcpp::SystemDefaultsQoS(),
        [this](const std::shared_ptr<geometry_msgs::msg::Twist> msg) -> void
        {
            RCLCPP_INFO(
            get_node()->get_logger(),
            "in subscriber = %.3f  rad/s | angular = %.3f  rad/s",
            msg->linear.x,
            msg->angular.z);
          if (!subscriber_is_active_)
          {
            RCLCPP_WARN(
              get_node()->get_logger(), "Can't accept new commands. subscriber is inactive");
            return;
          }
            RCLCPP_INFO(
            get_node()->get_logger(),
            "in subscriber = %.3f  rad/s | angular = %.3f  rad/s",
            msg->linear.x,
            msg->angular.z);
            // Write fake header in the stored stamped command
            auto twist_stamped = std::make_shared<Twist>();  // valid object
            //std::shared_ptr<Twist> twist_stamped; // creates null pointer
            twist_stamped->twist = *msg;
            twist_stamped->header.stamp = get_node()->get_clock()->now();
            received_velocity_msg_ptr_.writeFromNonRT(twist_stamped);
        });
        RCLCPP_INFO(
        get_node()->get_logger(),
        "subsriber-createdddddddddddddd");
    }
    // initialize odometry publisher and message
    odometry_publisher_ = get_node()->create_publisher<nav_msgs::msg::Odometry>(
    DEFAULT_ODOMETRY_TOPIC, rclcpp::SystemDefaultsQoS());
    realtime_odometry_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>>(
        odometry_publisher_);    

    // Append the tf prefix if there is one
    std::string tf_prefix = "";
    if (params_.tf_frame_prefix_enable)
    {
        if (params_.tf_frame_prefix != "")
        {
            tf_prefix = params_.tf_frame_prefix;
        }
        else
        {
            tf_prefix = std::string(get_node()->get_namespace());
        }

        // Make sure prefix does not start with '/' and always ends with '/'
        if (tf_prefix.back() != '/')
        {
            tf_prefix = tf_prefix + "/";
        }
        if (tf_prefix.front() == '/')
        {
            tf_prefix.erase(0, 1);
        }
    }
    const auto odom_frame_id = tf_prefix + params_.odom_frame_id;
    const auto base_frame_id = tf_prefix + params_.base_frame_id;
    
    // changing directly the msg stored in realtime_odometry_publisher_
    // reference to realtime_odometry_publisher_->msg_
    auto & odometry_message = realtime_odometry_publisher_->msg_;   
    odometry_message.header.frame_id = odom_frame_id;
    odometry_message.child_frame_id = base_frame_id;

    // limit the publication on the topics /odom and /tf
    publish_rate_ = params_.publish_rate;
    publish_period_ = rclcpp::Duration::from_seconds(1.0 / publish_rate_);
    // initialize odom values zeros
    odometry_message.twist =
        geometry_msgs::msg::TwistWithCovariance(rosidl_runtime_cpp::MessageInitialization::ALL);

    constexpr size_t NUM_DIMENSIONS = 6;
    for (size_t index = 0; index < 6; ++index)
    {
    // 0, 7, 14, 21, 28, 35
    const size_t diagonal_index = NUM_DIMENSIONS * index + index;
    odometry_message.pose.covariance[diagonal_index] = params_.pose_covariance_diagonal[index];
    odometry_message.twist.covariance[diagonal_index] = params_.twist_covariance_diagonal[index];
    } // ---? --> in update function we only update the linear and angular msg, covariance remains same
    
    // initialize transform publisher and message
    odometry_transform_publisher_ = get_node()->create_publisher<tf2_msgs::msg::TFMessage>(
    DEFAULT_TRANSFORM_TOPIC, rclcpp::SystemDefaultsQoS());
    realtime_odometry_transform_publisher_ =
    std::make_shared<realtime_tools::RealtimePublisher<tf2_msgs::msg::TFMessage>>(
        odometry_transform_publisher_);

    // keeping track of odom and base_link transforms only
    auto & odometry_transform_message = realtime_odometry_transform_publisher_->msg_;
    odometry_transform_message.transforms.resize(1);
    odometry_transform_message.transforms.front().header.frame_id = odom_frame_id;
    odometry_transform_message.transforms.front().child_frame_id = base_frame_id;

    previous_update_timestamp_ = get_node()->get_clock()->now();
    return controller_interface::CallbackReturn::SUCCESS;

}

bool CustomDiffController::reset(){
    odometry_.resetOdometry();
    reset_buffers();
    // release the old queue
    // std::queue<Twist> empty;
    // std::swap(previous_commands_, empty);

    registered_left_wheel_handles_.clear();
    registered_right_wheel_handles_.clear();

    subscriber_is_active_ = false;
    velocity_command_subscriber_.reset();
    velocity_command_unstamped_subscriber_.reset();
    is_halted = false;
    return true;
}

void CustomDiffController::reset_buffers(){

    // Empty out the old queue. Fill with zeros (not NaN) to catch early accelerations.
    Twist empty_twist(rosidl_runtime_cpp::MessageInitialization::ALL);
    empty_twist.twist.linear.x = 0.0;
    empty_twist.twist.angular.z = 0.0;
    // increate the counter to 1 for temporary pointer, 
    //then copy it to the thing_ then temp_ptr gets destroyed but we still we have a reference to the pointer through thing
    // received_velocity_msg_ptr_.set(std::make_shared<Twist>(empty_twist));    
    // // Fill last two commands with default constructed commands
    previous_commands_.emplace(empty_twist);
    previous_commands_.emplace(empty_twist);
    // std::queue<std::array<double, 2>> empty;
    // std::swap(previous_commands_, empty);
    // previous_commands_.push({{0.0, 0.0}});
    // previous_commands_.push({{0.0, 0.0}});
    // Fill RealtimeBuffer with NaNs so it will contain a known value
    // but still indicate that no command has yet been sent.
    received_velocity_msg_ptr_.reset();
    std::shared_ptr<Twist> empty_msg_ptr = std::make_shared<Twist>();
    empty_msg_ptr->header.stamp = get_node()->now();
    empty_msg_ptr->twist.linear.x = std::numeric_limits<double>::quiet_NaN();
    empty_msg_ptr->twist.linear.y = std::numeric_limits<double>::quiet_NaN();
    empty_msg_ptr->twist.linear.z = std::numeric_limits<double>::quiet_NaN();
    empty_msg_ptr->twist.angular.x = std::numeric_limits<double>::quiet_NaN();
    empty_msg_ptr->twist.angular.y = std::numeric_limits<double>::quiet_NaN();
    empty_msg_ptr->twist.angular.z = std::numeric_limits<double>::quiet_NaN();
    received_velocity_msg_ptr_.writeFromNonRT(empty_msg_ptr);

}


controller_interface::CallbackReturn CustomDiffController::on_activate(const rclcpp_lifecycle::State &){
    const auto left_result =
    configure_side("left", params_.left_wheel_names, registered_left_wheel_handles_);
    const auto right_result =
    configure_side("right", params_.right_wheel_names, registered_right_wheel_handles_);

    if (
    left_result == controller_interface::CallbackReturn::ERROR ||
    right_result == controller_interface::CallbackReturn::ERROR)
    {
    return controller_interface::CallbackReturn::ERROR;
    }

    if (registered_left_wheel_handles_.empty() || registered_right_wheel_handles_.empty())
    {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "Either left wheel interfaces, right wheel interfaces are non existent");
    return controller_interface::CallbackReturn::ERROR;
    }

    is_halted = false;
    subscriber_is_active_ = true;

    RCLCPP_DEBUG(get_node()->get_logger(), "Subscriber and publisher are now active.");
    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CustomDiffController::on_deactivate(const rclcpp_lifecycle::State &){
  subscriber_is_active_ = false;
  if (!is_halted)
  {
    halt();
    reset_buffers();
    is_halted = true;
  }
  registered_left_wheel_handles_.clear();
  registered_right_wheel_handles_.clear();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CustomDiffController::on_cleanup(const rclcpp_lifecycle::State &){
    if (!reset())
    {
        return controller_interface::CallbackReturn::ERROR;
    }

    return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn CustomDiffController::on_error(const rclcpp_lifecycle::State &){
    if (!reset())
    {
    return controller_interface::CallbackReturn::ERROR;
    }
    return controller_interface::CallbackReturn::SUCCESS;
}



controller_interface::CallbackReturn CustomDiffController::configure_side(const std::string & side, const std::vector<std::string> & wheel_names,std::vector<WheelHandle> & registered_handles){
  auto logger = get_node()->get_logger();

  if (wheel_names.empty())
  {
    RCLCPP_ERROR(logger, "No '%s' wheel names specified", side.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  // register handles
  registered_handles.reserve(wheel_names.size());
  for (const auto & wheel_name : wheel_names)
  {
    const auto interface_name = feedback_type();
    const auto state_handle = std::find_if(
      state_interfaces_.cbegin(), state_interfaces_.cend(),
      [&wheel_name, &interface_name](const auto & interface)
      {
        return interface.get_prefix_name() == wheel_name &&
               interface.get_interface_name() == interface_name;
      });

    if (state_handle == state_interfaces_.cend())
    {
      RCLCPP_ERROR(logger, "Unable to obtain joint state handle for %s", wheel_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }

    const auto command_handle = std::find_if(
      command_interfaces_.begin(), command_interfaces_.end(),
      [&wheel_name](const auto & interface)
      {
        return interface.get_prefix_name() == wheel_name &&
               interface.get_interface_name() == HW_IF_VELOCITY;
      });

    if (command_handle == command_interfaces_.end())
    {
      RCLCPP_ERROR(logger, "Unable to obtain joint command handle for %s", wheel_name.c_str());
      return controller_interface::CallbackReturn::ERROR;
    }

    registered_handles.emplace_back( WheelHandle{std::ref(*state_handle), std::ref(*command_handle)} );
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

// CustomDiffController::~CustomDiffController()
// {
// }

}  // namespace custom_diff_controller

#include "class_loader/register_macro.hpp"

CLASS_LOADER_REGISTER_CLASS(
  custom_diff_controller::CustomDiffController, controller_interface::ControllerInterface)
