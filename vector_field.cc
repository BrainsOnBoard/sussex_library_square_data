// OpenCV
#include <opencv2/opencv.hpp>

// BoB robotics 3rd party includes
#include "third_party/path.h"

// BoB robotics includes
#include "navigation/image_database.h"

// PSimpl includes
#include "psimpl.h"

// CLI11 includes
#include "CLI11.hpp"

#include "memory.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;
using namespace units::solid_angle;

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
void processRoute(const Navigation::ImageDatabase &database, double decimate,
                  cv::Mat &renderMatFull, cv::Mat &renderMatDecimated,
                  std::vector<cv::Point2f> &decimatedPoints)
{

    std::vector<float> routePointComponents;
    routePointComponents.reserve(database.size() * 2);

    {
        // Reserve temporary vector to hold route points, snapped to integer pixels
        std::vector<cv::Point2i> routePointPixels;
        routePointPixels.reserve(database.size());

        // Loop through route
        for(const auto &r : database) {
            // Get position of point in cm
            const centimeter_t x = r.position[0];
            const centimeter_t y = r.position[1];

            // Add x and y components of position to vector
            routePointComponents.emplace_back(x.value());
            routePointComponents.emplace_back(y.value());

            // Add x and y pixel values
            routePointPixels.emplace_back((int)std::round(x.value()), (int)std::round(y.value()));
        }

        // Build render matrix from route point pixels
        renderMatFull = cv::Mat(routePointPixels, true);
    }

    // Decimate route points
    std::vector<float> decimatedRoutePointComponents;
    psimpl::simplify_douglas_peucker<2>(routePointComponents.cbegin(), routePointComponents.cend(),
                                        decimate,
                                        std::back_inserter(decimatedRoutePointComponents));

    decimatedPoints.reserve(decimatedRoutePointComponents.size() / 2);

    {
        // Reserve temporary vector to hold decimated route points, snapped to integer pixels
        std::vector<cv::Point2i> decimatedPixels;
        decimatedPixels.reserve(decimatedRoutePointComponents.size() / 2);

        for(size_t i = 0; i < decimatedRoutePointComponents.size(); i += 2) {
            const float x = decimatedRoutePointComponents[i];
            const float y = decimatedRoutePointComponents[i + 1];

            decimatedPixels.emplace_back((int)std::round(x), (int)std::round(y));

            decimatedPoints.emplace_back(x, y);
        }

        // Build render matrix from decimated pixels
        renderMatDecimated = cv::Mat(decimatedPixels, true);
    }
}
//------------------------------------------------------------------------
// Get distance to route from point
std::tuple<centimeter_t, cv::Point2f, size_t, degree_t> getNearestPointOnRoute(const cv::Point2f &point,
                                                                               const std::vector<cv::Point2f> &routePoints)
{
    // Loop through points
    float shortestDistanceSquared = std::numeric_limits<float>::max();
    cv::Point2f nearestPoint;
    size_t nearestSegment;
    for(size_t i = 0; i < (routePoints.size() - 1); i++) {
        // Get vector pointing along segment and it's squared
        const cv::Point2f segmentVector = routePoints[i + 1] - routePoints[i];
        const float segmentLengthSquared = segmentVector.dot(segmentVector);

        // Get vector from start of segment to point
        const cv::Point2f segmentStartToPoint = point - routePoints[i];

        // Take dot product of two vectors and normalise, clamping at 0 and 1
        const float t = std::max(0.0f, std::min(1.0f, segmentStartToPoint.dot(segmentVector) / segmentLengthSquared));

        // Find nearest point on the segment
        const cv::Point2f nearestPointOnSegment = routePoints[i] + (t * segmentVector);

        // Get the vector from here to our point and hence the squared distance
        const cv::Point2f shortestSegmentToPoint = point - nearestPointOnSegment;
        const float distanceSquared = shortestSegmentToPoint.dot(shortestSegmentToPoint);

        // If this is shorter than current best, update current
        if(distanceSquared < shortestDistanceSquared) {
            shortestDistanceSquared = distanceSquared;
            nearestPoint = nearestPointOnSegment;
            nearestSegment = i;
        }
    }

    // Get vector in direction of nearest segment and hence heading
    const cv::Point2f nearestSegmentVector = routePoints[nearestSegment + 1] - routePoints[nearestSegment];
    const degree_t nearestSegmentHeading = radian_t(std::atan2(nearestSegmentVector.y, nearestSegmentVector.x));

    // Return shortest distance and position of nearest point
    return std::make_tuple(centimeter_t(std::sqrt(shortestDistanceSquared)), nearestPoint, nearestSegment, nearestSegmentHeading);
}
}   // Anonymous namespace

int main(int argc, char **argv)
{
    // Default command line arguments
    cv::Size imSize(120, 25);
    std::string routeName = "route5";
    std::string variantName = "skymask";
    std::string imageGridName = "mid_day";
    std::string outputImageName = "grid_image.png";
    std::string outputCSVName = "";
    std::string memoryType = "PerfectMemory";
    bool renderGoodMatches = true;
    bool renderBadMatches = false;
    bool renderRoute = true;
    bool renderDecimatedRoute = true;
    double fovDegrees = 90.0;
    double decimateDistance = 15.0;

    // Configure command line parser
    CLI::App app{"BoB robotics 'vector field' renderer"};
    app.add_option("--route", routeName, "Name of route", true);
    app.add_option("--grid", imageGridName, "Name of image grid", true);
    app.add_option("--variant", variantName, "Variant of route and grid to use", true);
    app.add_option("--width", imSize.width, "Width of unwrapped image", true);
    app.add_option("--height", imSize.height, "Height of unwrapped image", true);
    app.add_option("--output-image", outputImageName, "Name of output image to generate", true);
    app.add_option("--output-csv", outputCSVName, "Name of output CSV to generate", true);
    app.add_option("--decimate-distance", decimateDistance, "Threshold (in cm) for decimating route points", true);
    app.add_option("--fov", fovDegrees,
                   "For 'constrained' memories, what angle (in degrees) on either side of route should snapshots be matched in", true);
    app.add_set("--memory-type", memoryType, {"PerfectMemory", "PerfectMemoryConstrained", "InfoMax", "InfoMaxConstrained"},
                "Type of memory to use for navigation", true);
    /*app.add_flag("--render-good-matches,--no-render-good-matches{false}", renderGoodMatches,
                 "Should lines be rendered between grid points and 'good' matches");
    app.add_flag("--render-bad-matches,!--no-render-bad-matches", renderBadMatches,
                 "Should lines be rendered between grid points and 'bad' matches");
    app.add_flag("--render-route,!--no-render-route", renderRoute,
                 "Should unprocessed route be rendered");
    app.add_flag("--render-decimated-route,!--no-render-decimated-route", renderDecimatedRoute,
                 "Should decimated route be rendered");*/


    // Parse command line arguments
    CLI11_PARSE(app, argc, argv);

    // Create database from route
    const filesystem::path routePath = filesystem::path("routes") / routeName / variantName;
    std::cout << routePath << std::endl;
    Navigation::ImageDatabase route(routePath);

    std::unique_ptr<MemoryBase> memory;
    if(memoryType == "PerfectMemory") {
        memory.reset(new PerfectMemory(imSize, route,
                                       renderGoodMatches, renderBadMatches));
    }
    else if(memoryType == "PerfectMemoryConstrained") {
        memory.reset(new PerfectMemoryConstrained(imSize, route, degree_t(fovDegrees),
                                                  renderGoodMatches, renderBadMatches));
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

    // Process routes to get render images
    std::vector<cv::Point2f> decimatedRoutePoints;
    cv::Mat routePointsMat;
    cv::Mat decimatedRoutePointMat;
    processRoute(route, decimateDistance, routePointsMat, decimatedRoutePointMat, decimatedRoutePoints);

    // Load grid
    Navigation::ImageDatabase grid = filesystem::path("image_grids") /  imageGridName / variantName;
    assert(grid.isGrid());
    assert(grid.hasMetadata());

    // Read grid dimensions from meta data
    std::vector<double> size, seperationMM;
    grid.getMetadata()["grid"]["separationMM"] >> seperationMM;
    grid.getMetadata()["grid"]["size"] >> size;
    assert(size.size() == 3);
    assert(seperationMM.size() == 3);

    std::cout << size[0] << "x" << size[1] << " grid with " << seperationMM[0] << "x" << seperationMM[1] << "mm squares" << std::endl;

    // If a filename is specified, open CSV file other write to std::cout
    std::ofstream outputCSVFile;
    if(!outputCSVName.empty()) {
        outputCSVFile.open(outputCSVName);
    }
    std::ostream &outputCSV = outputCSVName.empty() ? std::cout : outputCSVFile;

    // Write header to CSV file
    memory->writeCSVHeader(outputCSV);
    outputCSV << std::endl;

    // Make a grid image with one pixel per cm
    cv::Mat gridImage((int)std::round(size[1] * seperationMM[1] * 0.1), 
                      (int)std::round(size[0] * seperationMM[0] * 0.1), 
                      CV_8UC3, cv::Scalar::all(0));

    // Draw route onto image
    if(renderRoute) {
        cv::polylines(gridImage, routePointsMat, false, CV_RGB(64, 64, 64));
    }

    if(renderDecimatedRoute) {
        cv::polylines(gridImage, decimatedRoutePointMat, false, CV_RGB(255, 255, 255));
    }

    // Loop through grid entries
    size_t numGridPointsWithinROI = 0;
    degree_squared_t sumSquareError = 0_sq_deg;
    for(const auto &g : grid) {
        const centimeter_t x = g.position[0];
        const centimeter_t y = g.position[1];

        // Get distance from grid point to route
        const auto nearestPoint = getNearestPointOnRoute(cv::Point2f(x.value(), y.value()), decimatedRoutePoints);

        // If snapshot is within R.O.I.
        if(std::get<0>(nearestPoint) < 4_m) {
            // Increment count
            numGridPointsWithinROI++;

            // Load snapshot and resize
            cv::Mat snapshot = g.load();
            cv::resize(snapshot, snapshot, imSize);

            // Test snapshot using memory
            memory->test(snapshot, g.heading, std::get<3>(nearestPoint));

            // Get magnitude of shortest angle between route and headig
            const degree_t angularError = shortestAngleBetween(memory->getBestHeading(), std::get<3>(nearestPoint));

            // Add to sum square error
            sumSquareError += (angularError * angularError);

            // Draw arrow showing vector field
            const centimeter_t xEnd = x + (60_cm * memory->getVectorLength() * cos(memory->getBestHeading()));
            const centimeter_t yEnd = y + (60_cm * memory->getVectorLength() * sin(memory->getBestHeading()));
            cv::arrowedLine(gridImage, cv::Point(x.value(), y.value()), cv::Point(xEnd.value(), yEnd.value()),
                            CV_RGB(0, 0, 255));

            // Write CSV line
            memory->writeCSVLine(outputCSV, x, y, angularError);
            outputCSV << std::endl;

            // Perform any memory-specific additional rendering
            memory->render(gridImage, x, y);

            // Update output image
            cv::imwrite(outputImageName, gridImage);
        }
    }

    std::cout << "RMSE:" << degree_t(sqrt(sumSquareError / (double)numGridPointsWithinROI)) << std::endl;


    return EXIT_SUCCESS;
}
