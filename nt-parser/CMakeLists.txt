PROJECT(dynet:nt-parser)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(nt-parser nt-parser.cc eval.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser dynet ${Boost_LIBRARIES} z)

ADD_EXECUTABLE(nt-parser-gen nt-parser-gen.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser-gen dynet ${Boost_LIBRARIES} z)

ADD_EXECUTABLE(nt-parser-gen-action-only nt-parser-gen-action-only.cc oracle.cc pretrained.cc)
target_link_libraries(nt-parser-gen-action-only dynet ${Boost_LIBRARIES} z)

