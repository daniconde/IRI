#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

const std::string datadir = "../../../data";


std::vector<cv::String> files;
std::map<cv::String, cv::RotatedRect> boxes;
RotatedRect currentBox;
int current_file;

Mat img; Mat img_gray;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);


void thresh_callback(int, void* );

void saveBox();
void loadFiles(const std::string &directory);
int selectImage(int index);
void mouseCallback(int event, int x, int y, int flags, void* userdata);

/** @function main */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    loadFiles(datadir);

    namedWindow( "Contours", CV_WINDOW_AUTOSIZE);
    setMouseCallback("Contours", mouseCallback);

    current_file = 0;
    selectImage(current_file);

    thresh_callback( 0, 0 );

    int key;
    while((key = waitKey()) != 27)
    {
        if (key == 'o')
        {
            saveBox();
            --current_file;
            if (current_file < 0)
                current_file = files.size() - 1;
            selectImage(current_file);
        }
        else if (key == 'p')
        {
            saveBox();
            ++current_file;
            if (current_file >= files.size())
                current_file = 0;
            selectImage(current_file);
        }
        
        if (key == 'q')
        {
            currentBox.angle = currentBox.angle - 0.2;
            thresh_callback(0, 0);
        }
        else if (key == 'e')
        {
            currentBox.angle = currentBox.angle + 0.2;
            thresh_callback(0, 0);
        }
        
        if (key == 'a')
        {
            currentBox.size.width -= 1.;
            thresh_callback(0, 0);
        }
        else if (key == 'd')
        {
            currentBox.size.width += 1;
            thresh_callback(0, 0);
        }
    }

    return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
    Mat img_final;
    img.copyTo(img_final);
    putText(img_final, files[current_file], Point(0, img_final.rows*0.95), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,0,255), 2, LINE_8, false);

    Point2f vertices[4];
    currentBox.points(vertices);
    for (int i = 0; i < 4; ++i)
    {
        line(img_final, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0), 2);
    }

    /// Show in a window
    imshow( "Contours", img_final);
}


cv::Point last(-1, -1);
void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        if (last.x == -1)
        {
            last = cv::Point(x, y);
        }
        else
        {
            cv::Point current(x, y);
            cv::Point center = (last + current) / 2;
            cv::Point diff = current - last;
            int distance = sqrt(pow(diff.x, 2) + pow(diff.y, 2));
            float diff_ratio = static_cast<float>(diff.y) / static_cast<float>(diff.x);
            float angle = atan(diff_ratio);
            currentBox.center = center;
            currentBox.size.height = distance;
            currentBox.angle = (angle / (2 * M_PI)) * 360 - 90;
            last = cv::Point(-1, -1);
        }
    }
    thresh_callback(0, 0);
}


void saveBox()
{
    auto it = boxes.find(files[current_file]);
    it->second = currentBox;
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

    int width = img.size().width;
    int height = img.size().height;
    RotatedRect defaultBox(cv::Point(width/2 - 25, height/2 - 25), cv::Size2f(50, 50), 0);
    auto it_pair = boxes.insert(
        std::pair<cv::String, RotatedRect>(files[current_file], defaultBox)
    );
    currentBox = it_pair.first->second;

    /// Convert image to gray and blur it
    thresh_callback(0, 0);

    return 1;
}

namespace bfs = boost::filesystem;
std::vector<bfs::path> img_extensions;

void loadFilesRec(const std::string &directory)
{
    vector<bfs::path> paths(0);
    copy(bfs::directory_iterator(directory), bfs::directory_iterator(), back_inserter(paths));
    for (auto file : paths)
    {
        if (bfs::is_regular_file(file.string()))
        {
            if (find(img_extensions.begin(), img_extensions.end(), file.extension()) != img_extensions.end())
            {
                cv::String filepath(file.string());
                files.push_back(filepath);
            }
        }
        else if (bfs::is_directory(file.string()))
        {
            loadFilesRec(file.string());
        }
    }
}

void loadFiles(const std::string &directory)
{
    img_extensions.push_back(bfs::path(".jpg"));
    img_extensions.push_back(bfs::path(".jpeg"));
    loadFilesRec(directory);
    img_extensions.clear();

    std::cout << files.size() << std::endl;
}
