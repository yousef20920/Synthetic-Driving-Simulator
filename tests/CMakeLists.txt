add_executable(sim_tests
  smoke_test.cpp
  world_grid_test.cpp
  actor_test.cpp
  simloop_test.cpp
  determinism_test.cpp
  bev_rasterizer_test.cpp
  noise_injector_test.cpp
  noise_config_loader_test.cpp
  metadata_exporter_test.cpp
  visual_verification_test.cpp
  car_behavior_test.cpp
  yielding_test.cpp
  pedestrian_behavior_test.cpp
  integration_behavior_test.cpp
)

target_link_libraries(sim_tests PRIVATE
  libsim
  GTest::gtest_main
)

target_compile_definitions(sim_tests PRIVATE
  SIM_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

set_target_properties(sim_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

include(GoogleTest)
gtest_discover_tests(sim_tests)
