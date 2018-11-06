#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace cv;
using namespace std;

const std::string datadir = "../../../data";

std::vector<cv::String> files;
int current_file;

Mat img; Mat img_gray; Mat img_final;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void* );

void loadFiles(const std::string &directory);
int selectImage(int index);

/** @function main */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    loadFiles(datadir);

    namedWindow( "Contours", CV_WINDOW_NORMAL);
    resizeWindow("Contours", 400, 400);

    current_file = 0;
    selectImage(current_file);


    createTrackbar( " Canny thresh:", "Contours", &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );

    int key;
    while((key = waitKey()) != 27)
    {
        if (key == 'a')
        {
            --current_file;
            selectImage(current_file);
        }
        else if (key == 'd')
        {
            ++current_file;
            selectImage(current_file);
        }
    }

    return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
    Mat canny_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    /// Detect edges using canny
    Canny( img_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    img.copyTo(img_final);
    /// Draw contours
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar(0, 255, 0);
        drawContours(img_final, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }

    /// Show in a window
    imshow( "Contours", img_final);
}


int selectImage(int index)
{
    index = index % files.size();

    if (files.size() == 0 || (img = imread(files[index])).empty())
    {
        if (files.size() == 0)
            std::cout << "No files to read" << std::endl;
        else
            std::cout << "Failed to load " << files[index] << std::endl;

        img = Mat(400, 400, CV_8UC3, Scalar(255, 0, 0));
        return 0;
    }

    /// Convert image to gray and blur it
    cvtColor( img, img_gray, CV_BGR2GRAY );
    blur( img_gray, img_gray, Size(3,3) );


    thresh_callback(0, 0);

    return 1;
}

void loadFiles(const std::string &directory)
{
    namespace bfs = boost::filesystem;
    vector<bfs::path> paths(0);
    copy(bfs::directory_iterator(directory), bfs::directory_iterator(), back_inserter(paths));
    for (auto file : paths)
    {
        if (bfs::is_regular_file(file.string()))
        {
            cv::String filepath(file.string());
            files.push_back(filepath);
        }
        else if (bfs::is_directory(file.string()))
        {
            loadFiles(file.string());
        }
    }
}
