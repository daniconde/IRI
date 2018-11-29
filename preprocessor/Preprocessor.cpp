#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <openssl/md5.h>


const std::string datadir = "../../../replace-select"; // <<<<<<<<<<<<<<<<<<<<<<<<< Tiene que ser igual para todos
const std::string datatrim = "../../../replace-select/trimmed"; // <<<<<<<<<<<<<<<<<<<<<<<<< Tiene que ser igual para todos
const std::string boxfile = "./boxes.txt";
const int batches = 1;
const int num_batch = 0; // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Diferente para cada uno (0, 1, 2)


std::vector<cv::String> files;
std::map<cv::String, cv::RotatedRect> boxes;
cv::RotatedRect originalBox, currentBox;
int current_file;

cv::Mat img;
int thresh = 100;
int max_thresh = 255;


cv::Point lastClick(-1, -1);
cv::Point rectP1(-1, -1), rectP2(-1, -1), rectP3(-1, -1);
int change_height = 1;
void draw();

void saveBox();
void loadFiles(const std::string &directory);
void saveTrimmedImages();
int selectImage(int index);
void mouseCallback(int event, int x, int y, int flags, void* userdata);

/** @function main */
int main( int argc, char** argv )
{
    loadFiles(datadir);

    cv::namedWindow( "Contours", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Contours", mouseCallback);

    int first_image = num_batch;

    current_file = first_image;
    selectImage(current_file);
    draw();

    int key;
    const float step_angle = 0.1;
    while((key = cv::waitKey()) != 27)
    {
        switch (key)
        {
            case 'o':
                saveBox();
                current_file -= batches;
                if (current_file < 0)
                {
                    if (files.size() % batches >= num_batch) 
                        current_file = (files.size() / batches) * batches + num_batch - 1;
                    else
                        current_file = (files.size() / batches) * batches - batches + num_batch - 1;
                }
                selectImage(current_file);
                break;
            case 'p':
                saveBox();
                current_file += batches;
                if (current_file >= files.size())
                    current_file = first_image;
                selectImage(current_file);
                break;
            case 'q':
                currentBox.angle = currentBox.angle - step_angle;
                break;
            case 'e':
                currentBox.angle = currentBox.angle + step_angle;
                break;
            case 'a':
                currentBox.size.width -= 1.;
                break;
            case 'd':
                currentBox.size.width += 1;
                break;
            case 'w':
                currentBox.size.height += 1;
                break;
            case 's':
                currentBox.size.height -= 1;
                break;
            case 'h':
                currentBox.center.x -= 1.;
                break;
            case 'k':
                currentBox.center.x += 1;
                break;
            case 'u':
                currentBox.center.y -= 1;
                break;
            case 'j':
                currentBox.center.y += 1;
                break;
            case 'r':
                currentBox = originalBox;
                lastClick = cv::Point(-1, -1);
                change_height = 1;
            case 'b':
                saveTrimmedImages();
        }
        draw();
    }

    return(0);
}


int scaleDown = 1;
void draw()
{
    cv::Mat imgFinal;
    img.copyTo(imgFinal);
    cv::putText(imgFinal, files[current_file], cv::Point(0, imgFinal.rows*0.95), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, cv::LINE_8, false);

    cv::Point2f vertices[4];
    currentBox.points(vertices);
    cv::line(imgFinal, vertices[0], vertices[1], cv::Scalar(0, 255, 0), 1);
    cv::line(imgFinal, vertices[1], vertices[2], cv::Scalar(0, 255, 0), 1);
    cv::line(imgFinal, vertices[2], vertices[3], cv::Scalar(0, 255, 0), 1);
    cv::line(imgFinal, vertices[3], vertices[0], cv::Scalar(0, 0, 255), 1);

    cv::line(
        imgFinal, 
        (vertices[0] + vertices[1]) / 2, 
        (vertices[2] + vertices[3]) / 2, 
        cv::Scalar(255, 255, 0),
        1
    );
    cv::line(
        imgFinal, 
        (vertices[1] + vertices[2]) / 2, 
        (vertices[3] + vertices[0]) / 2, 
        cv::Scalar(255, 255, 0),
        1
    );
    scaleDown = 1;

    while (imgFinal.size().width > 1700 || imgFinal.size().height > 900)
    {
        ++scaleDown;
        cv::resize(imgFinal, imgFinal, cv::Size2i(img.size() / scaleDown));
    }
    /// Show in a window
    cv::imshow( "Contours", imgFinal);
}


float distanceToLine(cv::Point p, cv::Point begin, cv::Point end)
{
    p = p - begin;
    end = end - begin;
    return p.cross(end) / norm(end);
}


cv::Point buttonDownClick(-1, -1);
void mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    x = x * scaleDown;
    y = y * scaleDown;
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        if (lastClick.x == -1)
        {
            lastClick = cv::Point(x, y);
        }
        else if (change_height == 1)
        {
            cv::Point currentClick(x, y);
            cv::Point center = (lastClick + currentClick) / 2;
            cv::Point diff = currentClick - lastClick;
            int distance = cv::norm(diff);
            float diff_ratio = static_cast<float>(diff.y) / static_cast<float>(diff.x);
            float angle = std::atan2(diff.y, diff.x);
            currentBox.center = center;
            currentBox.size.height = distance;
            currentBox.size.width = 0;
            currentBox.angle = (angle / (2 * M_PI)) * 360 + 90;
            lastClick = cv::Point(-1, -1);
            change_height = 0;
        }
        else
        {
            cv::Point currentClick(x, y);
            cv::Point center = (lastClick + currentClick) / 2;
            cv::Point diff = currentClick - lastClick;
            int distance = cv::norm(diff);
            currentBox.size.width = distance;
            lastClick = cv::Point(-1, -1);
            change_height = 1;
        }
    }
    else if (event == cv::EVENT_RBUTTONUP)
    {
        buttonDownClick = cv::Point(-1, -1);
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        buttonDownClick = cv::Point(x, y);
    }
    else if (event == cv::EVENT_MOUSEMOVE && buttonDownClick.x != -1)
    {
        cv::Point currentClick(x, y);
        cv::Point diff = currentClick - buttonDownClick;
        currentBox.center.x += diff.x;
        currentBox.center.y += diff.y;
        buttonDownClick = currentClick;
    }

    draw();
}


unsigned char img_md5[MD5_DIGEST_LENGTH];
char img_md5_str[MD5_DIGEST_LENGTH * 2];

void saveBox()
{
    if (originalBox.size == currentBox.size &&
        originalBox.center == currentBox.center &&
        originalBox.angle == currentBox.angle)
    {
        return;
    }

    unsigned char *data = img.data;
    int data_bytes = img.total() * img.elemSize();
    MD5(data, data_bytes, img_md5);
    for (auto i = 0; i < MD5_DIGEST_LENGTH; ++i)
    {
        img_md5_str[2*i] = "0123456789ABCDEF"[img_md5[i] / 16];
        img_md5_str[2*i+1] = "0123456789ABCDEF"[img_md5[i] % 16];
    }

    auto it = boxes.find(files[current_file]);
    it->second = currentBox;

    std::ofstream ofs;
    ofs.open(boxfile, std::ofstream::app);
    ofs << currentBox.center.x << " ";
    ofs << currentBox.center.y << " ";
    ofs << currentBox.size.width << " ";
    ofs << currentBox.size.height << " ";
    ofs << currentBox.angle << " ";
    for (const auto& c : img_md5_str)
    {
        ofs << c;
    }
    ofs << " ";
    ofs << std::string(files[current_file].c_str());
    ofs << std::endl;
}

int selectImage(int index)
{
    index = index % files.size();
    lastClick = cv::Point(-1, -1);
    change_height = 1;
    buttonDownClick = lastClick;

    if (files.size() == 0 || (img = cv::imread(files[index])).empty())
    {
        if (files.size() == 0)
            std::cout << "No files to read" << std::endl;
        else
            std::cout << "Failed to load " << files[index] << std::endl;

        img = cv::Mat(400, 400, CV_8UC3, cv::Scalar(255, 0, 0));
        return 0;
    }

    int width = img.size().width;
    int height = img.size().height;
    cv::RotatedRect defaultBox(cv::Point(0, 0), cv::Size2f(0, 0), 0);
    auto it_pair = boxes.insert(
        std::pair<cv::String, cv::RotatedRect>(files[current_file], defaultBox)
    );
    currentBox = it_pair.first->second;
    originalBox = currentBox;
    return 1;
}

namespace bfs = boost::filesystem;
std::vector<bfs::path> img_extensions;

void loadFilesRec(const std::string &directory)
{
    std::vector<bfs::path> paths(0);
    copy(bfs::directory_iterator(directory), bfs::directory_iterator(), std::back_inserter(paths));
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


void loadBoxes(const std::string &file)
{
    std::ifstream ifs;
    std::ofstream ofs;

    ifs.open(file);
    int a, b;
    std::string str;

    cv::RotatedRect box;
    while (ifs >> box.center.x)
    {
        ifs >> box.center.y;
        ifs >> box.size.width;
        ifs >> box.size.height;
        ifs >> box.angle;
        for (auto& c : img_md5_str)
        {
            ifs >> c;
        }
        ifs >> std::ws;
        std::getline(ifs, str);

        auto it_pair = boxes.insert(
            std::pair<cv::String, cv::RotatedRect>(str, box)
        );
        it_pair.first->second = box;
    }

    ifs.close();
}

void loadFiles(const std::string &directory)
{
    img_extensions.push_back(bfs::path(".jpg"));
    img_extensions.push_back(bfs::path(".jpeg"));
    loadFilesRec(directory);
    img_extensions.clear();

    if (!(files.size() > 0))
    {
        std::cout << "Wrong directory?" << std::endl;
        exit(1);
    }
    std::cout << "Loaded " << files.size() << " files" << std::endl;

    loadBoxes(boxfile);
}


void saveTrimmedImage()
{
    // rect is the RotatedRect (I got it from a contour...)
    // matrices we'll use
    
    float aspect_ratio = 90./160.;
    cv::Mat M, rotated, cropped;
    cv::RotatedRect rect = currentBox;
    // get angle and size from the bounding box
    float angle = rect.angle;
    cv::Size rect_size = rect.size;
    std::cout << rect_size << std::endl;
    if (rect_size.width == 0 || rect_size.height == 0)
        return;
    float rect_aspect_ratio = rect_size.width / rect_size.height;
    if (rect_aspect_ratio < aspect_ratio)
    {
        rect_size.width = rect_size.height * aspect_ratio;
    }
    else if (rect_aspect_ratio > aspect_ratio)
    {
        rect_size.height = rect_size.width / aspect_ratio;
    }
    // thanks to http://felix.abecassis.me/2011/10/opencv-rotation-deskewing/
    if (rect.angle < -45.) {
        angle += 90.0;
        std::swap(rect_size.width, rect_size.height);
    }
    angle += 180.0;
    // get the rotation matrix
    M = cv::getRotationMatrix2D(rect.center, angle, 1.0);
    // perform the affine transformation
    std::cout << img.size() << std::endl;
    cv::warpAffine(img, rotated, M, img.size(), cv::INTER_CUBIC);
    std::cout << rotated.size() << std::endl;
    // crop the resulting image
    cv::getRectSubPix(rotated, rect_size, rect.center, cropped);
    std::cout << cropped.size() << std::endl;

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(90, 160), 1., 1.);

    bfs::path filename(files[current_file].c_str());
    bfs::path trimdir(datatrim);
    bfs::path fullpath = trimdir / filename.filename();

    imwrite(fullpath.string(), resized);
}


void saveTrimmedImages()
{
    for (auto i = 0; i < files.size(); ++i)
    {
        current_file = i;
        bfs::path test(files[current_file]);
        std::cout << i << ": " << test.filename() << std::endl;
        selectImage(i);
        saveTrimmedImage();
    }
}
