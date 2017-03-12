
#include <iostream>
#include <string>
#include <cstdlib> // system()
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// mkdir
#include <sys/types.h>
#include <sys/stat.h>

enum Direction {
    up_down,
    left_right
};

cv::Mat GenerateImageWithSolidColor(const int& height, 
                                    const int& width, 
                                    const int& blueVal, 
                                    const int& greenVal, 
                                    const int& redVal) {
    cv::Mat img_(height, width, 0);
    img_.setTo(cv::Scalar(blueVal, greenVal, redVal));
    return img_;
}

void MaskImageAt(const int& row, const int& column, const int& newPixelValue, cv::Mat& image) {
    image.at<uchar>(row, column) = newPixelValue;
}

void CreateDirectory(const std::string& path) {
    mkdir(path.c_str(), 0775);
}

void CreateVideoFromImages(const std::string& videoName, 
                           const std::string& pathToImages, 
                           const std::string& imageBaseName, 
                           const int& frameHeight,
                           const int& frameWidth,
                           const int& imgCount,
                           const bool& reverse) {    
    auto start = 0;
    auto end = imgCount;
    auto step = 1;

    if (reverse) {
        start = imgCount;
        end = 0;
        step = -1;
    }

    
    double fps_ = 10;
    cv::Size frameSize_(frameHeight, frameWidth);
    cv::VideoWriter vw(videoName, CV_FOURCC('W', 'M', 'V', '2'), fps_, frameSize_);
    for (auto i = start; i != end; i += step) {
        auto currentImgName_ = pathToImages + "/" + imageBaseName + "-" + std::to_string(i) + ".jpg";
        cv::Mat img_ = cv::imread(currentImgName_, 1);
        vw.write(img_);
    }
}

void CleanVideosFolder() {
    std::string cleanFramesCommand_ = "exec rm -r ./videos/*";
    std::system(cleanFramesCommand_.c_str());
}

void DeleteFrames(const std::string& framesFolderName) {
    std::string cleanFramesCommand_ = "exec rm -rf " + framesFolderName;
    std::system(cleanFramesCommand_.c_str());
}

void GenerateBlackDotVideos(const Direction& dir, const int& size, const bool& reverse) {
    auto start = 0;        
    auto end = size;
    auto step = 1;
    
    std::string type;

    if (reverse) {
        start = size - 1;        
        end = -1;
        step = -1;
    }

    switch(dir) {
        case up_down:
            type = "up_down";
            if (reverse) type = "down_up";
        break;

        case left_right:
            type = "left_right";
            if (reverse) type = "right_left";
        break;
    }

    CreateDirectory("./videos/" + type); // Video type

    for (auto i = start; i != end; i += step) {
        // New video
        const auto videoFramesFolderName_ = "./videos/" + type + "/" + type + "-" + std::to_string(i);
        const auto videoName_ = "./videos/" + type + "/" + type + "-" + std::to_string(i) + ".mp4";
        CreateDirectory(videoFramesFolderName_);

        auto frameBaseName_ = "frame_" + std::to_string(i);

        for (auto j = start; j != end; j += step) {
            // White color is: (255, 255, 255)
            auto frame_ = GenerateImageWithSolidColor(size, size, 255, 255, 255);
            const auto blackColor_ = 0;

            const auto frameName_ = videoFramesFolderName_ + "/" + frameBaseName_ + "-" + std::to_string(j) + ".jpg";

            if (dir == Direction::left_right) {
                MaskImageAt(i, j, blackColor_, frame_);
            }
            else if (dir == Direction::up_down) {
                MaskImageAt(j, i, blackColor_, frame_);
            }

            cv::imwrite(frameName_, frame_);
        }
        CreateVideoFromImages(videoName_, videoFramesFolderName_, frameBaseName_, size, size, size, reverse);
    
        // Clean frames
        DeleteFrames(videoFramesFolderName_);
    }

}

int main() {
    // Clean whole /videos folder
    CleanVideosFolder();

    auto size_ = 100; // Images will be 100x100
    
    GenerateBlackDotVideos(Direction::up_down, size_, false);
    GenerateBlackDotVideos(Direction::up_down, size_, true); // Reverse

    GenerateBlackDotVideos(Direction::left_right, size_, false);
    GenerateBlackDotVideos(Direction::left_right, size_, true); // Reverse

    std::cout << "===Done===" << std::endl;
    return 0;
}
