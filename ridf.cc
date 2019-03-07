// OpenCV
#include <opencv2/opencv.hpp>

// BoB robotics 3rd party includes
#include "third_party/path.h"

// BoB robotics includes
#include "navigation/image_database.h"

// CLI11 includes
#include "CLI11.hpp"

#include "memory.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;
using namespace units::solid_angle;

int main(int argc, char **argv)
{
    // Default command line arguments
    cv::Size imSize(120, 25);
    std::string routeName = "route5";
    std::string variantName = "skymask";
    std::string outputImageName = "ridf_image.png";
    std::string outputCSVName = "";
    std::string memoryType = "PerfectMemory";
    std::string testImagePath;
    double fovDegrees = 90.0;

    // Configure command line parser
    CLI::App app{"BoB robotics R.I.D.F. renderer"};
    app.add_option("--test-image", testImagePath, "Path to image to test", false);
    app.add_option("--route", routeName, "Name of route", true);
    app.add_option("--variant", variantName, "Variant of route and grid to use", true);
    app.add_option("--width", imSize.width, "Width of unwrapped image", true);
    app.add_option("--height", imSize.height, "Height of unwrapped image", true);
    app.add_option("--output-image", outputImageName, "Name of output image to generate", true);
    app.add_option("--output-csv", outputCSVName, "Name of output CSV to generate", true);
    app.add_option("--fov", fovDegrees,
                   "For 'constrained' memories, what angle (in degrees) on either side of route should snapshots be matched in", true);
    app.add_set("--memory-type", memoryType, {"PerfectMemory", "PerfectMemoryConstrained", "InfoMax", "InfoMaxConstrained"},
                "Type of memory to use for navigation", true);

    // Parse command line arguments
    CLI11_PARSE(app, argc, argv);

    // Create database from route
    const filesystem::path routePath = filesystem::path("routes") / routeName / variantName;
    std::cout << routePath << std::endl;
    Navigation::ImageDatabase route(routePath);

    std::unique_ptr<MemoryBase> memory;
    if(memoryType == "PerfectMemory") {
        memory.reset(new PerfectMemory(imSize, route,
                                       false, false));
    }
    else if(memoryType == "PerfectMemoryConstrained") {
        memory.reset(new PerfectMemoryConstrained(imSize, route, degree_t(fovDegrees),
                                                  false, false));
    }
    else if(memoryType == "InfoMax") {
        memory.reset(new InfoMax(imSize, route));
    }
    else if(memoryType == "InfoMaxConstrained") {
        memory.reset(new InfoMaxConstrained(imSize, route, degree_t(fovDegrees)));
    }
    else {
        throw std::runtime_error("Memory type '" + memoryType + "' not supported");
    }


    // If a filename is specified, open CSV file other write to std::cout
    std::ofstream outputCSVFile;
    if(!outputCSVName.empty()) {
        outputCSVFile.open(outputCSVName);
    }
    std::ostream &outputCSV = outputCSVName.empty() ? std::cout : outputCSVFile;
    outputCSV << "Rotation[pixels], Rotation [degrees], familiarity" << std::endl;

    // Load test image and resize
    cv::Mat testImage = cv::imread(testImagePath, cv::IMREAD_GRAYSCALE);
    cv::resize(testImage, testImage, imSize);

    // Calculate RIDF from test image
    const auto ridf = memory->calculateRIDF(testImage);
    BOB_ASSERT(ridf.size() == (size_t)imSize.width);

    // Find minimum and maximum RIDF value
    const float maxRIDF = *std::max_element(ridf.cbegin(), ridf.cend());

    // Make an image to hold RIDF
    cv::Mat ridfImage(100, imSize.width * 10,
                      CV_8UC3, cv::Scalar::all(0));

    // Loop through RIDF columns
    for(size_t c = 0; c < ridf.size(); c++) {
        // Convert column into pixel rotation
        int pixelRotation = c;
        if(pixelRotation > (imSize.width / 2)) {
            pixelRotation -= imSize.width;
        }

        // Convert this into angle and write to output CSV
        const degree_t heading = turn_t((double)pixelRotation / (double)imSize.width);
        outputCSV << c << ", " << heading << ", " << ridf[c] << std::endl;

        // If this isn't the last column
        if(c < (ridf.size() - 1)) {
            // Get familiarity of this and next column
            const int familiarityPixels = 100 - (int)std::round(100.0 * (ridf[c] / maxRIDF));
            const int nextFamiliarityPixels = 100 - (int)std::round(100.0 * (ridf[c + 1] / maxRIDF));

            // Draw a line
            cv::line(ridfImage, cv::Point((int)(c * 10), familiarityPixels),
                    cv::Point((int)((c + 1) * 10), nextFamiliarityPixels),
                    CV_RGB(255, 255, 255));
        }
    }

    // Save RIDF image
    cv::imwrite(outputImageName, ridfImage);

    return EXIT_SUCCESS;
}
