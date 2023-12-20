# pragma once

#include <string>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <sys/time.h>

// Eigen includes
#include <Eigen/Dense>
// rapidjson includes
// SHOULD CHANGE THIS TOO UGLY
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>
// #include "/opt/homebrew/Cellar/rapidjson/1.1.0/include/rapidjson/document.h"
// #include "/opt/homebrew/Cellar/rapidjson/1.1.0/include/rapidjson/filereadstream.h"
// #include "/opt/homebrew/Cellar/rapidjson/1.1.0/include/rapidjson/filewritestream.h"
// #include "/opt/homebrew/Cellar/rapidjson/1.1.0/include/rapidjson/ostreamwrapper.h"
// #include "/opt/homebrew/Cellar/rapidjson/1.1.0/include/rapidjson/writer.h"
//Open3d
#include <open3d/geometry/PointCloud.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/geometry/Octree.h>
#include <open3d/io/PointCloudIO.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/geometry/BoundingVolume.h>
#include <open3d/geometry/KDTreeFlann.h>
#include <open3d/utility/Eigen.h>
#include <open3d/utility/Helper.h>
#include <open3d/utility/Console.h>
#include <open3d/visualization/utility/DrawGeometry.h>
#include <open3d/visualization/utility/ColorMap.h>


namespace rj = rapidjson;
using namespace std;
namespace fs = std::filesystem;


# define M_PI           3.14159265358979323846  /* pi */
typedef open3d::geometry::PointCloud PointCloud;
typedef open3d::geometry::TriangleMesh TriangleMesh;
typedef open3d::geometry::Octree Octree;
typedef open3d::geometry::OctreeNode OctreeNode;
typedef open3d::geometry::OctreeNodeInfo OctreeNodeInfo;
typedef open3d::geometry::OctreeLeafNode OctreeLeafNode;
typedef open3d::geometry::OctreePointColorLeafNode OctreePointColorLeafNode;

inline bool read_config(
    char* path, 
    rj::Document & config_doc)
{
    FILE *config_fp = fopen(path, "rb");

    if (!config_fp)
    {
        cerr << "Error: unable to open argv[1]" << endl;
        return false;
    }

    char config_readBuffer[65536];
    rj::FileReadStream config_is(config_fp, config_readBuffer,
                                 sizeof(config_readBuffer));
    config_doc.ParseStream(config_is);
    if (config_doc.HasParseError())
    {
        cerr << "Error: failed to parse JSON document" << endl;
        fclose(config_fp);
        return false;
    }
    fclose(config_fp);
    return true;
}

inline void show_progress_bar(
    const char* name, size_t current, size_t total)
{
    cout << "processing " << name << ": [";
    size_t current_i = current / total * 10;
    // for (size_t i = 0; i < current_i; i++)
    //     cout << "#";
    // for (size_t i = 9; i >= current_i; i--)
    //     cout << " ";
    cout << "]" << current << "/" << total << endl;
}