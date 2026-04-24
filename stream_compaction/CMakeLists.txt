set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "thrust.h"
    "radix.h"
    "shared.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "thrust.cu"
    "radix.cu"
    "shared.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers} PREFIX stream_compaction)
source_group(Sources FILES ${sources} PREFIX stream_compaction)

add_library(stream_compaction ${sources} ${headers})

target_include_directories(stream_compaction PUBLIC ${CMAKE_SOURCE_DIR})

if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES native)
endif()

target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda>")

target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;--extended-lambda>")

message(NOTICE "Info: Added stream_compaction library.")