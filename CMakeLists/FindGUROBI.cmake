if (NOT GUROBI_FOUND)
    # 硬编码查找头文件和链接库
    set(SEARCH_PATHS_FOR_HEADERS
        "$ENV{GUROBI_HOME}/include"
    )
    set(SEARCH_PATHS_FOR_LIBRARIES
        "$ENV{GUROBI_HOME}/lib"
    )
    find_path(GUROBI_INCLUDE_DIR gurobi_c++.h
      PATHS ${SEARCH_PATHS_FOR_HEADERS}
    )
    find_library( GUROBI_C_LIBRARY
                  NAMES libgurobi120.so
                  PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                  )
 
    find_library( GUROBI_CXX_LIBRARY_DEBUG
                NAMES libgurobi_c++.a
                PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                )
 
    find_library( GUROBI_CXX_LIBRARY_RELEASE
                NAMES libgurobi_c++.a
                PATHS ${SEARCH_PATHS_FOR_LIBRARIES}
                )
 
    # setup header file directories
    set(GUROBI_INCLUDE_DIRS ${GUROBI_INCLUDE_DIR})
 
    # setup libraries files
    set(GUROBI_LIBRARIES
            debug ${GUROBI_CXX_LIBRARY_DEBUG}
            optimized ${GUROBI_CXX_LIBRARY_RELEASE}
            ${GUROBI_C_LIBRARY}
            )
 
endif()
 
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_INCLUDE_DIRS)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARIES)
 
mark_as_advanced(GUROBI_LIBRARIES GUROBI_INCLUDE_DIRS GUROBI_INCLUDE_DIR)