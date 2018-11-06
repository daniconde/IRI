cmake_minimum_required(VERSION 2.8)

project( IRI )

find_package( OpenCV 3.4.1 REQUIRED )
find_package( Boost REQUIRED COMPONENTS filesystem )

add_executable( Preprocessor.x Preprocessor.cpp )

target_link_libraries( Preprocessor.x ${OpenCV_LIBS} )
target_link_libraries( Preprocessor.x ${Boost_LIBRARIES} )
