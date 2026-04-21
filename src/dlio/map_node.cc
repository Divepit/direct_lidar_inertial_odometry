/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include <thread>
#ifdef __GLIBC__
  #include <malloc.h>
#endif

#include "dlio/map.h"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char** argv) {
#ifdef __GLIBC__
  // Reduce glibc heap arena fragmentation when many threads are used
  mallopt(M_ARENA_MAX, 1);
#endif

  rclcpp::init(argc, argv);
  auto node = std::make_shared<dlio::MapNode>();
  std::weak_ptr<dlio::MapNode> weak_node(node);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), /* threads */ 2);

  executor.add_node(node);

  rclcpp::on_shutdown([weak_node, &executor]() {
    if (auto node = weak_node.lock()) {
      node->requestStop();
    }
    executor.cancel();
  });

  executor.spin();

  node->requestStop();
  executor.remove_node(node);
  node.reset();

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
