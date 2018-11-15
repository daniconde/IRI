Required libraries:

    - OpenCV (and dependencies: vtk, glew, hdf5)
    - Boost
    - OpenSSL

How to build:

    Inside the preprocessor directory:
        
        - In Preprocessor.cpp, set the constants datadir and boxfile.

        - Then:

            $ mkdir Debug
            $ cd Debug
            $ cmake .. -DCMAKE_BUILD_TYPE=Debug
            $ make

Controls:

    >>Disable Caps Lock!<<

    Define a rectangle by left clicking the midpoints of the top and bottom segments (important order) and
    then the midpoints of the side segments (any order).

    Move the rectangle by dragging with right mouse button.

    Fine move the rectangle with the keys U, H, J and K.

    Fine adjust the rectangle size with the keys W, A, S and D.

    Fine adjust the rectangle angle with Q and E.

    Press R to reload the original rectangle.
