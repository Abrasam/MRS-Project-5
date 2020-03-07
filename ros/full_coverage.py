







def run(args):
    rospy.init_mode('full_coverage')

    # Update control at 100Hz
    rate_limiter = rospy.Rate(100)
    publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

    previous_time = rospy.Time.now()

    i = 0
    while i < 10 and not rospy.is_shutdown():
        publisher.publish(stop_msg)
        rate_limiter.sleep()
        i += 1

    while not rospy.is_shutdown():
        current_time = rospy.Time.now().to_sec()

        





if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs Full Coverage naviation')
  args, unknown = parser.parse_known_args()
  try:
    run(args)
  except rospy.ROSInterruptException:
    pass
