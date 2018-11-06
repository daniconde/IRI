Required libraries:

    OpenCV (and dependencies: vtk, glew, hdf5)
    Boost

How to build:

    Inside the preprocessor directory:
        
        - In Preprocessor.cpp, set the constant datadir with the path to the data directory.

        - Then:

            $ mkdir Debug
            $ cd Debug
            $ cmake .. -DCAMKE_BUILD_TYPE=Debug
            $ make


