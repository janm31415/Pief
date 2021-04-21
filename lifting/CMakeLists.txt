
set(HDRS
lifting_api.h
lifting.h
sobolev.h
)
	
set(SRCS
sobolev.cpp
)

set(CMAKE_C_FLAGS_DEBUG "/W4 /MP /GF /RTCu /Od /MDd /Zi")
set(CMAKE_CXX_FLAGS_DEBUG "/W4 /MP /GF /RTCu /Od /MDd /Zi")
set(CMAKE_C_FLAGS_RELEASE "/W4 /MP /GF /O2 /Ob2 /Oi /Ot /MD /Zi /DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "/W4 /MP /GF /O2 /Ob2 /Oi /Ot /MD /Zi /DNDEBUG")

include ("../jtk/jtk/jtk.cmake")

# general build definitions
add_definitions(-D_UNICODE)
add_definitions(-DUNICODE)
add_definitions(-D_SCL_SECURE_NO_WARNINGS)

add_library(lifting SHARED ${HDRS} ${SRCS})
source_group("Header Files" FILES ${HDRS})
source_group("Source Files" FILES ${SRCS})

target_include_directories(lifting
    PRIVATE
	  ${CMAKE_CURRENT_SOURCE_DIR}/..
    )	
	
target_link_libraries(lifting
    PRIVATE	
    )	