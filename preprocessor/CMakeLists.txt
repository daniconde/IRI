cmake_minimum_required(VERSION 2.8)

project( IRI )

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS filesystem )
find_package( OpenSSL REQUIRED )

add_executable( Preprocessor.x Preprocessor.cpp )

target_link_libraries( Preprocessor.x ${OpenCV_LIBS} )
target_link_libraries( Preprocessor.x ${Boost_LIBRARIES} )
target_link_libraries( Preprocessor.x ${OPENSSL_CRYPTO_LIBRARY} )
