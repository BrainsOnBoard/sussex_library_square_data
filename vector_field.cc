// OpenCV
#include <opencv2/opencv.hpp>

// BoB robotics 3rd party includes
#include "third_party/path.h"

// BoB robotics includes
#include "navigation/image_database.h"
#include "navigation/infomax.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_store_raw.h"

// PSimpl includes
#include "psimpl.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;
using namespace units::solid_angle;

//#define INFOMAX
#define CONSTRAIN_HEADINGS

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
degree_t shortestAngleBetween(degree_t x, degree_t y)
{
    return atan2(sin(x - y), cos(x - y));
}
//------------------------------------------------------------------------
void processRoute(const Navigation::ImageDatabase &database,
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
                                        15.0,
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
//------------------------------------------------------------------------
template<typename FloatType>
void writeWeights(const BoBRobotics::Navigation::InfoMax<FloatType> &infomax, const filesystem::path &path)
{
    // Write weights to disk
    const auto weights = infomax.getWeights();
    std::ofstream netFile(path.str(), std::ios::binary);
    const int size[2] { (int) weights.rows(), (int) weights.cols() };
    netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
    netFile.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(FloatType));
}
//------------------------------------------------------------------------
template<typename FloatType>
auto readData(const filesystem::path &weightPath)
{
    // Open file
    std::ifstream is(weightPath.str(), std::ios::binary);
    if (!is.good()) {
        throw std::runtime_error("Could not open " + weightPath.str());
    }

    // The matrix size is encoded as 2 x int32_t
    int32_t size[2];
    is.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Create data array and fill it
    Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> data(size[0], size[1]);
    is.read(reinterpret_cast<char *>(data.data()), sizeof(FloatType) * data.size());

    return std::move(data);
}
//------------------------------------------------------------------------
template<typename FloatType>
Navigation::InfoMaxRotater<> createInfoMax(const cv::Size &imSize, const filesystem::path &weightPath, const Navigation::ImageDatabase &route)
{
     if(weightPath.exists()) {
        std::cout << "Loading weights from " << weightPath << std::endl;
        const auto weights = readData<FloatType>(weightPath);

        Navigation::InfoMaxRotater<> infomax(imSize, weights);
        return std::move(infomax);
    }
    else {
        Navigation::InfoMaxRotater<> infomax(imSize);
        infomax.trainRoute(route, true);
        writeWeights(infomax, weightPath.str());
        std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
        return std::move(infomax);
    }
}
//------------------------------------------------------------------------
std::tuple<degree_t, size_t, float> getConstrainedHeading(const std::vector<std::vector<float>> &allDifferences,
                                                          degree_t gridHeading, degree_t routeHeading, degree_t fov,
                                                          const cv::Size &imSize)
{
    // **NOTE** this currently uses a super-naive approach because good solution is non-trivial as
    // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle

    /*// Get angles on either side of route within which snapshots will be matched
    const degree_t routeA = routeHeading - fov;
    const degree_t routeB = routeHeading + fov;

    // Get shortest angle between each of these route angles and grid
    const degree_t gridRotationA = shortestAngleBetween(gridHeading, routeA);
    const degree_t gridRotationB = shortestAngleBetween(gridHeading, routeB);

    // Convert these into pixel rotations
    int columnA = (int)std::round((turn_t(gridRotationA) * (double)imSize.width).value());
    int columnB = (int)std::round((turn_t(gridRotationB) * (double)imSize.width).value());

    // If either pixel rotation is negative (
    if(columnA < 0) {
        columnA += imSize.width;
    }
    if(columnB < 0) {
        columnB += imSize.width;
    }

    // Get min and max
    const int minColumn = std::min(columnA, columnB);
    const int maxColumn = std::max(columnA, columnB);
    assert(minColumn >= 0);
    assert(maxColumn >= 0);
    assert(minColumn < imSize.width);
    assert(maxColumn < imSize.width);*/

    // Loop through snapshots
    float lowestDifference = std::numeric_limits<float>::max();
    size_t bestSnapshot = std::numeric_limits<size_t>::max();
    degree_t bestRotation = 0_deg;
    for(size_t i = 0; i < allDifferences.size(); i++) {
        const auto &snapshotDifferences = allDifferences[i];

        // Loop through acceptable range of columns
        for(int c = 0; c < imSize.width; c++) {
            // If this snapshot is a better match than current best
            if(snapshotDifferences[c] < lowestDifference) {
                // Convert column into pixel rotation
                int pixelRotation = c;
                if(pixelRotation > (imSize.width / 2)) {
                    pixelRotation -= imSize.width;
                }

                // Convert this into angle
                const degree_t rotation = turn_t((double)pixelRotation / (double)imSize.width);

                // If the distance between this angle from grid and route angle is within FOV
                if(fabs(shortestAngleBetween(rotation + gridHeading, routeHeading)) < fov) {
                    // Update best
                    bestSnapshot = i;
                    bestRotation = rotation;
                    lowestDifference = snapshotDifferences[c];
                }
            }
        }
    }

    // Check valid snapshot actually exists
    assert(bestSnapshot != std::numeric_limits<size_t>::max());

    // Scale difference to match code in ridf_processors.h:57
    return std::make_tuple(bestRotation, bestSnapshot, lowestDifference / 255.0f);
}
}   // Anonymous namespace

int main()
{
    const filesystem::path routePath("routes/route5/skymask");

    const cv::Size imSize(120, 25);

    // Create database from route
    Navigation::ImageDatabase route(routePath);

    // Default algorithm: find best-matching snapshot, use abs diff
#ifdef INFOMAX
    const auto pm = createInfoMax<float>(imSize, routePath / "infomax.bin", route);
#else
    Navigation::PerfectMemoryRotater<> pm(imSize);
    pm.trainRoute(route, true);
    std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
#endif

    // Process routes to get render images
    std::vector<cv::Point2f> decimatedRoutePoints;
    cv::Mat routePointsMat;
    cv::Mat decimatedRoutePointMat;
    processRoute(route, routePointsMat, decimatedRoutePointMat, decimatedRoutePoints);

    // Load grid
    Navigation::ImageDatabase grid("image_grids/mid_day/skymask");
    assert(grid.isGrid());
    assert(grid.hasMetadata());

    // Read grid dimensions from meta data
    std::vector<double> size, seperationMM;
    grid.getMetadata()["grid"]["separationMM"] >> seperationMM;
    grid.getMetadata()["grid"]["size"] >> size;
    assert(size.size() == 3);
    assert(seperationMM.size() == 3);

    std::cout << size[0] << "x" << size[1] << " grid with " << seperationMM[0] << "x" << seperationMM[1] << "mm squares" << std::endl;

    // Make a grid image with one pixel per cm
    cv::Mat gridImage((int)std::round(size[1] * seperationMM[1] * 0.1), 
                      (int)std::round(size[0] * seperationMM[0] * 0.1), 
                      CV_8UC3, cv::Scalar::all(0));

    // Draw route onto image
    cv::polylines(gridImage, routePointsMat, false, CV_RGB(64, 64, 64));
    cv::polylines(gridImage, decimatedRoutePointMat, false, CV_RGB(255, 255, 255));

    // Loop through grid entries
#ifdef INFOMAX
    std::vector<float> allDifferences;
#else
    bool showBadMatches = false;
#ifndef CONSTRAIN_HEADINGS
    std::vector<std::vector<float>> allDifferences;
#endif
#endif

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
            cv::Mat snapshot = g.loadGreyscale();
            cv::resize(snapshot, snapshot, imSize);

            // Get best heading using perfect memory
            degree_t bestHeading;
            float lowestDifference;
#ifdef INFOMAX
            std::tie(bestHeading, lowestDifference, allDifferences) = pm.getHeading(snapshot);
            const double vectorLength = 1.0f;
#else
            size_t bestSnapshotIndex;
#ifdef CONSTRAIN_HEADINGS
            // Get 'matrix' of differences from perfect memory
            const auto &allDifferences = pm.getImageDifferences(snapshot);


            // From these find the best snapshot within 90 degrees of route at nearest point
            std::tie(bestHeading, bestSnapshotIndex, lowestDifference) = getConstrainedHeading(allDifferences,
                                                                                               g.heading, std::get<3>(nearestPoint), 90_deg,
                                                                                               imSize);

            // Check resultant heading is actually within 90 degrees of route at nearest point
            assert(fabs(degree_t(atan2(sin(std::get<3>(nearestPoint) - bestHeading), cos(std::get<3>(nearestPoint) - bestHeading)))));
#else
            std::tie(bestHeading, bestSnapshotIndex, lowestDifference, allDifferences) = pm.getHeading(snapshot);
#endif
            const double vectorLength = (1.0 - lowestDifference);
#endif

            // Add grid image heading to best heading as that is what it's relative to
            bestHeading += g.heading;

            // Get magnitude of shortest angle between route and headig
            const degree_t angularError = shortestAngleBetween(bestHeading, std::get<3>(nearestPoint));

            // Add to sum square error
            sumSquareError += (angularError * angularError);

            // Draw arrow showing vector field
            const centimeter_t xEnd = x + (60_cm * vectorLength * cos(bestHeading));
            const centimeter_t yEnd = y + (60_cm * vectorLength * sin(bestHeading));
            cv::arrowedLine(gridImage, cv::Point(x.value(), y.value()), cv::Point(xEnd.value(), yEnd.value()),
                            CV_RGB(0, 0, 255));

            std::cout << "(" << x << ", " << y << ") : " << bestHeading << "(" << angularError << "), " << lowestDifference;
#ifndef INFOMAX
            std::cout << ", " << bestSnapshotIndex;

            // Get position of best snapshot
            const centimeter_t bestSnapshotX = route[bestSnapshotIndex].position[0];
            const centimeter_t bestSnapshotY = route[bestSnapshotIndex].position[1];

            //cv::line(gridImage, cv::Point(x.value(), y.value()), std::get<1>(nearestPoint), CV_RGB(0, 255, 0));

            // If snapshot is less than 3m away i.e. algorithm hasn't entirely failed draw line from snapshot to route
            const bool goodMatch = (sqrt(((bestSnapshotX - x) * (bestSnapshotX - x)) + ((bestSnapshotY - y) * (bestSnapshotY - y))) < 3_m);
            if(goodMatch || showBadMatches) {
                cv::line(gridImage, cv::Point(x.value(), y.value()), cv::Point(bestSnapshotX.value(), bestSnapshotY.value()),
                        goodMatch ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0));
            }
#endif
            std::cout << std::endl;


            cv::imwrite("grid_image.png", gridImage);
        }
    }

    std::cout << "RMSE:" << degree_t(sqrt(sumSquareError / (double)numGridPointsWithinROI)) << std::endl;


    return EXIT_SUCCESS;
}
