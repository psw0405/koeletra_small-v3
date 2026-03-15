cmake_minimum_required(VERSION 3.20)
project(ner_onnx_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set -DONNXRUNTIME_DIR=<path-to-onnxruntime>.")
endif()

add_executable(ner_onnx_infer ner_infer.cpp)

target_include_directories(ner_onnx_infer PRIVATE "${ONNXRUNTIME_DIR}/include")

if(WIN32)
  target_link_directories(ner_onnx_infer PRIVATE "${ONNXRUNTIME_DIR}/lib")
  target_link_libraries(ner_onnx_infer PRIVATE onnxruntime)
else()
  target_link_libraries(ner_onnx_infer PRIVATE "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()
