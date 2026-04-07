# libsim: core simulation static library.
# Sources are added here as they are created in subsequent plans.
add_library(libsim STATIC
  sim_placeholder.cpp
  WorldGrid.cpp
  Scene.cpp
  CsvLogger.cpp
  SimLoop.cpp
  BevRasterizer.cpp
  NoiseInjector.cpp
  NoiseConfigLoader.cpp
  MetadataExporter.cpp
)

target_include_directories(libsim
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(libsim PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
