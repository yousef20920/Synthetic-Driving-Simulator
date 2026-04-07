add_executable(sim_runner main.cpp)

target_link_libraries(sim_runner PRIVATE libsim)

target_compile_definitions(sim_runner PRIVATE
  SIM_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
