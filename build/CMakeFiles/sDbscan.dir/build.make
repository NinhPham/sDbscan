# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/npha145/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/npha145/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build"

# Include any dependencies generated for this target.
include CMakeFiles/sDbscan.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sDbscan.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sDbscan.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sDbscan.dir/flags.make

CMakeFiles/sDbscan.dir/src/main.cpp.o: CMakeFiles/sDbscan.dir/flags.make
CMakeFiles/sDbscan.dir/src/main.cpp.o: /home/npha145/Dropbox\ (Uni\ of\ Auckland)/Working/_Code/C++/sDbscan/src/main.cpp
CMakeFiles/sDbscan.dir/src/main.cpp.o: CMakeFiles/sDbscan.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sDbscan.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sDbscan.dir/src/main.cpp.o -MF CMakeFiles/sDbscan.dir/src/main.cpp.o.d -o CMakeFiles/sDbscan.dir/src/main.cpp.o -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/main.cpp"

CMakeFiles/sDbscan.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sDbscan.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/main.cpp" > CMakeFiles/sDbscan.dir/src/main.cpp.i

CMakeFiles/sDbscan.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sDbscan.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/main.cpp" -o CMakeFiles/sDbscan.dir/src/main.cpp.s

CMakeFiles/sDbscan.dir/src/Utilities.cpp.o: CMakeFiles/sDbscan.dir/flags.make
CMakeFiles/sDbscan.dir/src/Utilities.cpp.o: /home/npha145/Dropbox\ (Uni\ of\ Auckland)/Working/_Code/C++/sDbscan/src/Utilities.cpp
CMakeFiles/sDbscan.dir/src/Utilities.cpp.o: CMakeFiles/sDbscan.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sDbscan.dir/src/Utilities.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sDbscan.dir/src/Utilities.cpp.o -MF CMakeFiles/sDbscan.dir/src/Utilities.cpp.o.d -o CMakeFiles/sDbscan.dir/src/Utilities.cpp.o -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/Utilities.cpp"

CMakeFiles/sDbscan.dir/src/Utilities.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sDbscan.dir/src/Utilities.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/Utilities.cpp" > CMakeFiles/sDbscan.dir/src/Utilities.cpp.i

CMakeFiles/sDbscan.dir/src/Utilities.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sDbscan.dir/src/Utilities.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/Utilities.cpp" -o CMakeFiles/sDbscan.dir/src/Utilities.cpp.s

CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o: CMakeFiles/sDbscan.dir/flags.make
CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o: /home/npha145/Dropbox\ (Uni\ of\ Auckland)/Working/_Code/C++/sDbscan/src/sDbscan.cpp
CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o: CMakeFiles/sDbscan.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o -MF CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o.d -o CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/sDbscan.cpp"

CMakeFiles/sDbscan.dir/src/sDbscan.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sDbscan.dir/src/sDbscan.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/sDbscan.cpp" > CMakeFiles/sDbscan.dir/src/sDbscan.cpp.i

CMakeFiles/sDbscan.dir/src/sDbscan.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sDbscan.dir/src/sDbscan.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/sDbscan.cpp" -o CMakeFiles/sDbscan.dir/src/sDbscan.cpp.s

CMakeFiles/sDbscan.dir/src/fast_copy.c.o: CMakeFiles/sDbscan.dir/flags.make
CMakeFiles/sDbscan.dir/src/fast_copy.c.o: /home/npha145/Dropbox\ (Uni\ of\ Auckland)/Working/_Code/C++/sDbscan/src/fast_copy.c
CMakeFiles/sDbscan.dir/src/fast_copy.c.o: CMakeFiles/sDbscan.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/sDbscan.dir/src/fast_copy.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/sDbscan.dir/src/fast_copy.c.o -MF CMakeFiles/sDbscan.dir/src/fast_copy.c.o.d -o CMakeFiles/sDbscan.dir/src/fast_copy.c.o -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fast_copy.c"

CMakeFiles/sDbscan.dir/src/fast_copy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/sDbscan.dir/src/fast_copy.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fast_copy.c" > CMakeFiles/sDbscan.dir/src/fast_copy.c.i

CMakeFiles/sDbscan.dir/src/fast_copy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/sDbscan.dir/src/fast_copy.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fast_copy.c" -o CMakeFiles/sDbscan.dir/src/fast_copy.c.s

CMakeFiles/sDbscan.dir/src/fht.c.o: CMakeFiles/sDbscan.dir/flags.make
CMakeFiles/sDbscan.dir/src/fht.c.o: /home/npha145/Dropbox\ (Uni\ of\ Auckland)/Working/_Code/C++/sDbscan/src/fht.c
CMakeFiles/sDbscan.dir/src/fht.c.o: CMakeFiles/sDbscan.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/sDbscan.dir/src/fht.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/sDbscan.dir/src/fht.c.o -MF CMakeFiles/sDbscan.dir/src/fht.c.o.d -o CMakeFiles/sDbscan.dir/src/fht.c.o -c "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fht.c"

CMakeFiles/sDbscan.dir/src/fht.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/sDbscan.dir/src/fht.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fht.c" > CMakeFiles/sDbscan.dir/src/fht.c.i

CMakeFiles/sDbscan.dir/src/fht.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/sDbscan.dir/src/fht.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/src/fht.c" -o CMakeFiles/sDbscan.dir/src/fht.c.s

# Object files for target sDbscan
sDbscan_OBJECTS = \
"CMakeFiles/sDbscan.dir/src/main.cpp.o" \
"CMakeFiles/sDbscan.dir/src/Utilities.cpp.o" \
"CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o" \
"CMakeFiles/sDbscan.dir/src/fast_copy.c.o" \
"CMakeFiles/sDbscan.dir/src/fht.c.o"

# External object files for target sDbscan
sDbscan_EXTERNAL_OBJECTS =

sDbscan: CMakeFiles/sDbscan.dir/src/main.cpp.o
sDbscan: CMakeFiles/sDbscan.dir/src/Utilities.cpp.o
sDbscan: CMakeFiles/sDbscan.dir/src/sDbscan.cpp.o
sDbscan: CMakeFiles/sDbscan.dir/src/fast_copy.c.o
sDbscan: CMakeFiles/sDbscan.dir/src/fht.c.o
sDbscan: CMakeFiles/sDbscan.dir/build.make
sDbscan: CMakeFiles/sDbscan.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable sDbscan"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sDbscan.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sDbscan.dir/build: sDbscan
.PHONY : CMakeFiles/sDbscan.dir/build

CMakeFiles/sDbscan.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sDbscan.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sDbscan.dir/clean

CMakeFiles/sDbscan.dir/depend:
	cd "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan" "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan" "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build" "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build" "/home/npha145/Dropbox (Uni of Auckland)/Working/_Code/C++/sDbscan/build/CMakeFiles/sDbscan.dir/DependInfo.cmake" "--color=$(COLOR)"
.PHONY : CMakeFiles/sDbscan.dir/depend

