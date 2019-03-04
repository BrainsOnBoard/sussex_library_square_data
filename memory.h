#pragma once

// BoB robotics 3rd party includes
#include "third_party/units.h"

// BoB robotics includes
#include "navigation/image_database.h"
#include "navigation/infomax.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_store_raw.h"

inline units::angle::degree_t shortestAngleBetween(units::angle::degree_t x, units::angle::degree_t y)
{
    return units::math::atan2(units::math::sin(x - y), units::math::cos(x - y));
}

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
class MemoryBase
{
public:
    MemoryBase();

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) = 0;
    virtual void writeCSVHeader(std::ostream &os);
    virtual void writeCSVLine(std::ostream &os, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY, units::angle::degree_t angularError);
    virtual void render(cv::Mat &, units::length::centimeter_t, units::length::centimeter_t)
    {
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    units::angle::degree_t getBestHeading() const{ return m_BestHeading; }
    float getLowestDifference() const{ return m_LowestDifference; }
    float getVectorLength() const{ return m_VectorLength; }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void setBestHeading(units::angle::degree_t bestHeading){ m_BestHeading = bestHeading; }
    void setLowestDifference(float lowestDifference){ m_LowestDifference = lowestDifference; }
    void setVectorLength(float vectorLength){ m_VectorLength = vectorLength; }

private:
    //-----------------------------------------------------------------------units-
    // Members
    //------------------------------------------------------------------------
    units::angle::degree_t m_BestHeading;
    float m_LowestDifference;
    float m_VectorLength;
};

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
class PerfectMemory : public MemoryBase
{
public:
    PerfectMemory(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route,
                  bool renderGoodMatches, bool renderBadMatches);

    //------------------------------------------------------------------------
    // MemoryBase virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t) override;
    virtual void writeCSVHeader(std::ostream &os);
    virtual void writeCSVLine(std::ostream &os, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY, units::angle::degree_t angularError);
    virtual void render(cv::Mat &image, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY);

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    size_t getBestSnapshotIndex() const{ return m_BestSnapshotIndex; }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const BoBRobotics::Navigation::PerfectMemoryRotater<> &getPM() const{ return m_PM; }

    void setBestSnapshotIndex(size_t bestSnapshotIndex){ m_BestSnapshotIndex = bestSnapshotIndex; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    BoBRobotics::Navigation::PerfectMemoryRotater<> m_PM;
    const BoBRobotics::Navigation::ImageDatabase &m_Route;
    size_t m_BestSnapshotIndex;
    const bool m_RenderGoodMatches;
    const bool m_RenderBadMatches;
};

//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
class PerfectMemoryConstrained : public PerfectMemory
{
public:
    PerfectMemoryConstrained(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route, units::angle::degree_t fov,
                             bool renderGoodMatches, bool renderBadMatches);


    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) override;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const units::angle::degree_t m_FOV;
    const int m_ImageWidth;
};

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
class InfoMax : public MemoryBase
{
    using InfoMaxType = BoBRobotics::Navigation::InfoMaxRotater<BoBRobotics::Navigation::InSilicoRotater, float>;
    using InfoMaxWeightMatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

public:
    InfoMax(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route);

    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t) override;

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const InfoMaxType &getInfoMax() const{ return m_InfoMax; }

private:
    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    // **TODO** move into BoB robotics
    static void writeWeights(const InfoMaxWeightMatrixType &weights, const filesystem::path &weightPath);
    // **TODO** move into BoB robotics
    static InfoMaxWeightMatrixType readWeights(const filesystem::path &weightPath);
    static InfoMaxType createInfoMax(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route);

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    InfoMaxType m_InfoMax;
};

//------------------------------------------------------------------------
// InfoMaxConstrained
//------------------------------------------------------------------------
class InfoMaxConstrained : public InfoMax
{
public:
    InfoMaxConstrained(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route, units::angle::degree_t fov);

    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) override;

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const units::angle::degree_t m_FOV;
    const int m_ImageWidth;
};
