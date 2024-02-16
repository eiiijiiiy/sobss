# pragma once

#include <iostream>
#include <string>
#include <map>
#include <filesystem>
#include <sys/time.h>
#include <set>
#include <vector>
#include <iomanip>
#include <random>
#include <iterator>
#include <algorithm>

// rapidjson includes
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

// nlopt includes
#include <nlopt.hpp>

// xtensor includes
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>

//Open3d includes
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

// gurobi includes
#include <gurobi_c++.h>

// CGAL includes
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/repair_degeneracies.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/draw_polyhedron.h>
#include <CGAL/draw_nef_3.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <boost/iterator/function_output_iterator.hpp>

// openmp includes
#include <omp.h>

// eigen includes
#include <Eigen/Dense>

using namespace std;
namespace rj = rapidjson;


enum EnumNormalGroup {horizontal=0, non_horizontal=1, other=2}; // normal group

typedef open3d::geometry::PointCloud                PointCloud;
typedef open3d::geometry::TriangleMesh              TriangleMesh;
typedef open3d::geometry::Octree                    Octree;
typedef open3d::geometry::OctreeNode                OctreeNode;
typedef open3d::geometry::OctreeNodeInfo            OctreeNodeInfo;
typedef open3d::geometry::OctreeLeafNode            OctreeLeafNode;
typedef open3d::geometry::OctreePointColorLeafNode  OctreePointColorLeafNode;

typedef CGAL::Exact_predicates_exact_constructions_kernel     K;
typedef K::Point_3                                            Point_3;
typedef K::Vector_3                                           Vector_3;
typedef CGAL::Surface_mesh<Point_3>                           Mesh;
typedef CGAL::Polyhedron_3<K>                                 Polyhedron;
typedef CGAL::Nef_polyhedron_3<K>                             Nef_Polyhedron;
typedef Mesh::Vertex_index                                    vertex_descriptor;
typedef Mesh::Face_index                                      face_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;


bool read_config(
    const char* path, 
    rj::Document& config_doc)
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

void ClusterPointNormals(
    const PointCloud& pcd, 
    PointCloud& h_pcd,
    PointCloud& nh_pcd,
    vector<EnumNormalGroup> &labels, // why we also need this enumeric group?
    double h_z_thresh, 
    double v_z_thresh)
{ 
    h_pcd.Clear();
    nh_pcd.Clear();

    for (size_t i = 0; i < pcd.points_.size(); i++) {
        if (abs(pcd.normals_[i][2]) >= h_z_thresh) {
            labels[i] = EnumNormalGroup::horizontal;
            h_pcd.points_.push_back(pcd.points_[i]);
            h_pcd.normals_.push_back(pcd.normals_[i]);
        }
        else if ((abs(pcd.normals_[i][2]) <= v_z_thresh) ||
                    (pcd.normals_[i][2] > 0)){
            labels[i] = EnumNormalGroup::non_horizontal;
            nh_pcd.points_.push_back(pcd.points_[i]);
            nh_pcd.normals_.push_back(pcd.normals_[i]);
        }
    }    
}

typedef struct{
    vector<double> radian_frequency;
    double interval;
    int half_PI_bin_num;
    double orientation;
    double lower_bound;
    double upper_bound;
    int max_iter;

    double evaluate()
    {
        auto tar_bin_float = orientation/interval;    // a virtual bin in float, range = [0, half_PI_bin_num * 2]
        double error = 0;

        for (size_t i = 0; i < half_PI_bin_num; i++)
        {
            // abs diff in radian
            auto rot_diff = abs(tar_bin_float - i) * interval;
            // if closer than +/- pi/4 ==> 
            error += cos(rot_diff) * sin(rot_diff) * radian_frequency[i]; 
        }
        return error;
    }

}OrientationPcd;


static double evaluate_nlopt (const vector<double> &x, vector<double> &grad, void *data)
{
    OrientationPcd* ori_pcd = (OrientationPcd*) data;
    ori_pcd->orientation = x[0];
    return ori_pcd->evaluate();
}


int dfo_solve(
    OrientationPcd& ori_pcd)
{
    vector<double> x(1);
    x[0] = ori_pcd.orientation; 
    double minf;
    nlopt::srand(20210509);
    // Other NLopt solvers: GN_DIRECT, LN_COBYLA, LN_NELDERMEAD, LN_NEWUOA_BOUND, LN_BOBYQA
    nlopt::opt opt(nlopt::algorithm::GN_DIRECT, 1);   // dimension = 1
    opt.set_lower_bounds(ori_pcd.lower_bound);
    opt.set_upper_bounds(ori_pcd.upper_bound);
    opt.set_maxeval(ori_pcd.max_iter);
    opt.set_min_objective(evaluate_nlopt, &ori_pcd);
    opt.set_xtol_rel(1e-6);
    cout << "nlopt model set" << endl;
    try
    {
        cout << "begin to optimize ... \n";
        nlopt::result result = opt.optimize(x, minf);
        ori_pcd.orientation = x[0];
        cout << "optimized \n";
        return EXIT_SUCCESS;
    }
    catch(exception &e) 
    {
        cout << "NLOpt failed: " << e.what() << endl;
        return EXIT_FAILURE;
    }
}


double EstimateHorizontalOrientation(
                    PointCloud &vertical_pcd,
                    int half_PI_bin_num,
                    int max_iter = 100){
    // compute the average horizontal normal direction
    size_t num_point = vertical_pcd.points_.size();
    cout << "* profile of axis aligning: "    
         << "\t bin number of the pi/2 = " << half_PI_bin_num
         << "\t max iteration = " << max_iter
         << "\t point number of the vertical point cloud = " << num_point
         << "\n";

    vector<double> normal_radian_frequency(half_PI_bin_num, 0.0);
    double interval = M_PI / (half_PI_bin_num * 2);

    for(size_t i = 0; i < num_point; i++) {
        Eigen::Vector2f horizontal_normal = Eigen::Vector2f(vertical_pcd.normals_[i][0], 
                                    vertical_pcd.normals_[i][1]).normalized();
        double radian = atan2(horizontal_normal[1], horizontal_normal[0]);
        int bin_id = floor(radian / interval);
        bin_id = bin_id - floor( (double)bin_id / half_PI_bin_num) * half_PI_bin_num;
        normal_radian_frequency[bin_id] += (double)1/num_point;
    }
    cout << "* normal_radian_frequency computed." << endl;
    OrientationPcd ori_pcd {
        .radian_frequency = normal_radian_frequency,
        .interval = interval,
        .half_PI_bin_num = half_PI_bin_num,
        .orientation = M_PI/4,
        .lower_bound = 0,
        .upper_bound = M_PI/2,
        .max_iter = max_iter
    };
    cout << "Orientation pcd initiated." << endl;

    // solve
    if (dfo_solve(ori_pcd) == EXIT_SUCCESS)
    {
        cout << "DFO succeeded" << endl;
        double orientation = ori_pcd.orientation;
        return orientation;
    }
    else
    {
        cerr << "Error: DFO failed " << endl;
        return -100.0;
    }
}


void group_normals(
    const vector<Eigen::Vector2d> &normals,
    const vector<Eigen::Vector2d> &group_target_normals,
    vector<size_t> &group_id,
    vector<vector<size_t>> &groups){
    
    size_t num_normals = normals.size();
    size_t num_targets = group_target_normals.size();
    double best_dist, this_dist;
    for(size_t i = 0 ; i < num_normals; i++){
        best_dist = 1e10;
        for(size_t t = 0; t < num_targets; t++){
            this_dist = (normals[i] - group_target_normals[t]).norm();
            if(this_dist < best_dist){
                best_dist = this_dist;
                group_id[i] = t;
            }
        }
        groups[group_id[i]].push_back(i);
    }
}


void fast_vote(
    PointCloud &v_pcd,
    double voxel_size,
    // vector<size_t> &group_id,
    vector<size_t> &primary_ptids,
    string output_voted_param_path, 
    string output_voted_center_pcd_path,
    double normal_pair_thresh = 0.1,
    double normal_vertical_z_thresh = 0.5){
    
    double min_pcd_z = v_pcd.GetMinBound()(2);
    double normal_vertical_thresh = 
        sqrt(1 - normal_vertical_z_thresh * normal_vertical_z_thresh);

    size_t primary_num_pts = primary_ptids.size();
    vector<bool> find_vote(primary_num_pts, false);
    vector<vector<double>> voted_centers(primary_num_pts, vector<double>{0,0,0});
    vector<double> voted_dimensions(primary_num_pts, 0);
    vector<double> voted_horizontal_normal_norm(primary_num_pts, 0);
    vector<double> voted_z(primary_num_pts, 0);
    vector<vector<double>> pair_1(primary_num_pts, vector<double>{0,0,0});
    vector<vector<double>> pair_2(primary_num_pts, vector<double>{0,0,0});

    struct timeval t_start, t_end;
    long sec, usec;
    double timecost;
    
    xt::xarray<double> transformed_pts(vector<size_t>{primary_num_pts, 3});
    xt::xarray<double> transformed_normals(vector<size_t>{primary_num_pts, 3});
    xt::xarray<size_t> transformed_pts_voxel_idx(vector<size_t>{primary_num_pts, 3});
    vector<array<size_t, 3>> positive_n_transformed_pts_voxel_idx_, 
                            negative_n_transformed_pts_voxel_idx_;
    size_t sample_size = 0;
 
    // gettimeofday(&t_start, NULL);

    double orientation = atan2(1, 0);
    vector<array<size_t, 3>> n_transformed_pts_voxel_idx_(primary_ptids.size());

    for(size_t vn = 0; vn < primary_ptids.size(); vn++){
        transformed_pts(vn, 0) = v_pcd.points_[primary_ptids[vn]][0];
        transformed_pts(vn, 1) = v_pcd.points_[primary_ptids[vn]][1];
        transformed_pts(vn, 2) = v_pcd.points_[primary_ptids[vn]][2];
        // (0,x,x)
        transformed_normals(vn, 0) = v_pcd.normals_[primary_ptids[vn]][0];
        transformed_normals(vn, 1) = v_pcd.normals_[primary_ptids[vn]][1];
        transformed_normals(vn, 2) = v_pcd.normals_[primary_ptids[vn]][2];
    }
    // gettimeofday(&t_end, NULL);
    // sec = t_end.tv_sec - t_start.tv_sec;
    // usec = t_end.tv_usec - t_start.tv_usec;
    // timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    // process_t.push_back(timecost);

    gettimeofday(&t_start, NULL);
    xt::xarray<double> MIN, MAX;
    xt::xarray<size_t> BIN_NUM;
    MIN = xt::amin(transformed_pts, {0});
    MAX = xt::amax(transformed_pts, {0});
    BIN_NUM = xt::ceil((MAX - MIN)/voxel_size);

    xt::col(transformed_pts_voxel_idx, 0) = xt::floor((xt::col(transformed_pts, 0) - MIN[0]) / voxel_size);
    xt::col(transformed_pts_voxel_idx, 1) = xt::floor((xt::col(transformed_pts, 1) - MIN[1]) / voxel_size);
    xt::col(transformed_pts_voxel_idx, 2) = xt::floor((xt::col(transformed_pts, 2) - MIN[2]) / voxel_size);

    PointCloud ppcd = open3d::geometry::PointCloud();
    PointCloud npcd = open3d::geometry::PointCloud();
    xt::xarray<size_t> voxel_n_pts = xt::zeros<size_t>(BIN_NUM); 
    xt::xarray<double> voxel_pn_pts = xt::zeros<double>(BIN_NUM); 

    for(size_t vn = 0; vn < primary_ptids.size(); vn++){
        n_transformed_pts_voxel_idx_[vn][0] = transformed_pts_voxel_idx(vn, 0);
        n_transformed_pts_voxel_idx_[vn][1] = transformed_pts_voxel_idx(vn, 1);
        n_transformed_pts_voxel_idx_[vn][2] = transformed_pts_voxel_idx(vn, 2);
        voxel_n_pts(
            n_transformed_pts_voxel_idx_[vn][0],
            n_transformed_pts_voxel_idx_[vn][1],
            n_transformed_pts_voxel_idx_[vn][2]) = 1;
        // after axis-aligned rotation, the primary normal is fixed to (0,1)
        double costheta = transformed_normals(vn, 0) * 0
                        + transformed_normals(vn, 1) * 1; 
        double hor_norm = sqrt(
                pow(transformed_normals(vn, 0), 2) 
                + pow(transformed_normals(vn, 1), 2));
        if (costheta >= 0){
            // positive_n_transformed_pts_voxel_idx_.push_back(n_transformed_pts_voxel_idx_[vn]);
            ppcd.points_.push_back(v_pcd.points_[primary_ptids[vn]]);
            voxel_pn_pts(
                n_transformed_pts_voxel_idx_[vn][0],
                n_transformed_pts_voxel_idx_[vn][1],
                n_transformed_pts_voxel_idx_[vn][2]) = hor_norm;
        }else{
            // negative_n_transformed_pts_voxel_idx_.push_back(n_transformed_pts_voxel_idx_[vn]);
            npcd.points_.push_back(v_pcd.points_[primary_ptids[vn]]);
            voxel_pn_pts(
                n_transformed_pts_voxel_idx_[vn][0],
                n_transformed_pts_voxel_idx_[vn][1],
                n_transformed_pts_voxel_idx_[vn][2]) = -hor_norm;
        }   
    }

    // auto npi = xt::index_view(voxel_n_pts, n_transformed_pts_voxel_idx_);
    // npi += 1;
    // auto pnpi = xt::index_view(voxel_pn_pts, positive_n_transformed_pts_voxel_idx_);
    // pnpi = 1;
    // auto nnpi = xt::index_view(voxel_pn_pts, negative_n_transformed_pts_voxel_idx_);
    // nnpi = -1;

    cout << " voxel_n_pts sum : "<< xt::sum(voxel_n_pts) << endl;

    // gettimeofday(&t_end, NULL);
    // sec = t_end.tv_sec - t_start.tv_sec;
    // usec = t_end.tv_usec - t_start.tv_usec;
    // timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    // process_t.push_back(timecost);

    /*****************SAVE THE PRIMARY NORMAL GROUP *******************/
    // <--speed test, do not count the time for saving results
    // string positive_pcd_group_path = normal_group_prefix + "_p.ply";
    // string negative_pcd_group_path = normal_group_prefix + "_n.ply";
    // open3d::io::WritePointCloud(positive_pcd_group_path, ppcd);
    // open3d::io::WritePointCloud(negative_pcd_group_path, npcd);
    // speed test, do not count the time for saving results -->
    /******************************************************************/
    // gettimeofday(&t_start, NULL);
    // open3d::utility::ConsoleProgressBar sample_progress_bar(
    //     primary_ptids.size(), "voting inside a normal group", true);

    size_t num_vertical=0, num_incline=0;

    # pragma omp parallel for num_threads(16) 
    for(size_t p = 0; p < primary_num_pts; p++){ 
        // show_progress_bar("voting", p, primary_ptids.size());
        size_t b1, b21, h;
        double x, y, z, dim, y1, y2;
        b1 = n_transformed_pts_voxel_idx_[p][0];
        b21 = n_transformed_pts_voxel_idx_[p][1];
        h = n_transformed_pts_voxel_idx_[p][2];
        double this_pn = voxel_pn_pts(b1, b21, h);
        /************FIND b12**************/
        if (this_pn < 0)
        {
            auto b21_ = xt::view(voxel_n_pts, b1, xt::range(b21+1, BIN_NUM(1)), h);
            if(xt::sum(b21_)[0]>0){
                auto b21_temp = xt::flatten_indices(xt::argwhere(b21_>0));
                auto b22_ = b21_temp(0) + b21 + 1;
                double b22__pn =  voxel_pn_pts(b1, b22_, h);
                if((b22__pn > 0) && (abs(b22__pn + this_pn) < normal_pair_thresh))
                {
                    x = double(b1 * voxel_size) + MIN[0];
                    y1 = double(b21 * voxel_size) + MIN[1];
                    y2 = double(b22_ * voxel_size) + MIN[1];
                    y = (y1 + y2) / 2;
                    z = double(h * voxel_size) + MIN[2];
                    dim = double((b22_ - b21) * voxel_size) / 2;
                    sample_size++;
                    find_vote[p] = true;
                    if (b22__pn > normal_vertical_thresh) // vertical
                    {   
                        voted_centers[p][0] = x;
                        voted_centers[p][1] = y;
                        voted_centers[p][2] = z;
                        voted_dimensions[p] = dim;
                        voted_horizontal_normal_norm[p] = b22__pn;
                        voted_z[p] = 9999.0;
                        pair_1[p][0] = x;
                        pair_1[p][1] = y1;
                        pair_1[p][2] = z;
                        pair_2[p][0] = x;
                        pair_2[p][1] = y2;
                        pair_2[p][2] = z;
                        num_vertical++;
                    }
                    else // incline
                    {
                        double n_v = sqrt(1 - pow(b22__pn, 2));
                        double ball_r = dim / b22__pn;
                        double ball_z = z - ball_r * n_v;
                        double zc = ball_z + ball_r / n_v;

                        assert (zc < min_pcd_z + 200);
                        voted_centers[p][0] = x;
                        voted_centers[p][1] = y;
                        voted_centers[p][2] = ball_z;
                        voted_dimensions[p] = ball_r;
                        voted_horizontal_normal_norm[p] = b22__pn;
                        voted_z[p] = zc;
                        pair_1[p][0] = x;
                        pair_1[p][1] = y1;
                        pair_1[p][2] = z;
                        pair_2[p][0] = x;
                        pair_2[p][1] = y2;
                        pair_2[p][2] = z;
                        num_incline++;
                    }
                }
            }
        }
        else // this_pn >= 0
        {
            auto _b21 = xt::view(voxel_n_pts, b1, xt::range(0, b21), h);
            if(xt::sum(_b21)[0] > 0){
                auto b21_temp = xt::flatten_indices(xt::argwhere(_b21>0));
                auto b_22 = b21_temp(b21_temp.size()-1);
                double b_22_pn = voxel_pn_pts(b1, b_22, h);
                if((b_22_pn < 0) && (abs(b_22_pn + this_pn) < normal_pair_thresh)){
                    x = double(b1 * voxel_size) + MIN[0];
                    y1 = double(b21 * voxel_size) + MIN[1];
                    y2 = double(b_22 * voxel_size) + MIN[1];
                    y = (y1 + y2) / 2;
                    z = double(h * voxel_size) + MIN[2];
                    dim = double((b21 - b_22) * voxel_size) / 2;
                    // cout << "dim: " << dim << ", b21: " << b21 << ", b_22: "<< b_22 << endl;
                    sample_size++;
                    find_vote[p] = true;
                    if (this_pn > normal_vertical_thresh)
                    {
                        voted_centers[p][0] = x;
                        voted_centers[p][1] = y;
                        voted_centers[p][2] = z;
                        voted_dimensions[p] = dim;
                        voted_horizontal_normal_norm[p] = this_pn;
                        voted_z[p] = 9999.0;
                        pair_1[p][0] = x;
                        pair_1[p][1] = y1;
                        pair_1[p][2] = z;
                        pair_2[p][0] = x;
                        pair_2[p][1] = y2;
                        pair_2[p][2] = z;
                        num_vertical++;
                    }
                    else // incline
                    {
                        double n_v = sqrt(1 - pow(this_pn, 2));
                        double ball_r = dim / this_pn;
                        double ball_z = z - ball_r * n_v;
                        double zc = ball_z + ball_r / n_v;
                        // cout << "dim: " << dim
                        //      << "zc: " << zc << ", min pcd z: " << min_pcd_z 
                        //      << "this pn" << this_pn << ", nv: " << n_v << endl;
                        assert (zc < min_pcd_z + 200);
                        voted_centers[p][0] = x;
                        voted_centers[p][1] = y;
                        voted_centers[p][2] = ball_z;
                        voted_dimensions[p] = ball_r;
                        voted_horizontal_normal_norm[p] = this_pn;
                        voted_z[p] = zc;
                        pair_1[p][0] = x;
                        pair_1[p][1] = y1;
                        pair_1[p][2] = z;
                        pair_2[p][0] = x;
                        pair_2[p][1] = y2;
                        pair_2[p][2] = z;
                        num_incline++;
                    }
                }
            } 
        }
    }
    cout << " \t num vertical: " << num_vertical << ", num incline: " << num_incline << endl;
    // gettimeofday(&t_end, NULL);
    // sec = t_end.tv_sec - t_start.tv_sec;
    // usec = t_end.tv_usec - t_start.tv_usec;
    // timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    // process_t.push_back(timecost);

    /************SAVE SKELETON ATOMS OF THE PRIMARY NORMAL GROUP ***************/
    // <--speed test, do not count the time for saving results
    std::ofstream voted_params_out(output_voted_param_path);
    PointCloud voted_center_pcd = open3d::geometry::PointCloud();

    for (size_t i = 0; i <primary_num_pts; i ++){
        if (find_vote[i])
        {
            voted_params_out<< voted_centers[i][0] << " " 
                        << voted_centers[i][1] << " " 
                        << voted_centers[i][2] << " " 
                        << voted_dimensions[i] << " " 
                        << voted_horizontal_normal_norm[i] << " " 
                        << voted_z[i] << " " 
                        << pair_1[i][0] << " "
                        << pair_1[i][1] << " "
                        << pair_1[i][2] << " "
                        << pair_2[i][0] << " "
                        << pair_2[i][1] << " "
                        << pair_2[i][2] << " "
                        << endl;
            voted_center_pcd.points_.push_back(Eigen::Vector3d(
                voted_centers[i][0],voted_centers[i][1],voted_centers[i][2]));
        }
    }
    open3d::io::WritePointCloud(output_voted_center_pcd_path, voted_center_pcd);
    cout << "* saved solid voted centers to " << output_voted_center_pcd_path << endl;
    cout << "* saved solid voted params to " << output_voted_param_path << endl;
}


void get_8pts_of_a_volume_from_param(
    const vector<double>& param,
    Eigen::MatrixXd& matrix_8pts)
{
    matrix_8pts.resize(8, 3);
    double nv = sqrt(1 - param[6] * param[6]);
    double z_top = param[2] + param[5] * nv + param[4] * param[6] / 2;
    double z_btm = param[2] + param[5] * nv - param[4] * param[6] / 2;
    double y_front_top = param[1] + param[5] * param[6] - param[4] / 2 * nv;
    double y_front_btm = param[1] + param[5] * param[6] + param[4] / 2 * nv;
    double y_back_top = param[1] - param[5] * param[6] + param[4] / 2 * nv;
    double y_back_btm = param[1] - param[5] * param[6] - param[4] / 2 * nv;
    double x_right = param[0] + param[3] / 2;
    double x_left = param[0] - param[3] / 2;

    // flt
    matrix_8pts.row(0) = Eigen::Vector3d(x_left, y_front_top, z_top).transpose(); 
    // frt
    matrix_8pts.row(1) = Eigen::Vector3d(x_right, y_front_top, z_top).transpose();
    // flb
    matrix_8pts.row(2) = Eigen::Vector3d(x_left, y_front_btm, z_btm).transpose();
    // frb
    matrix_8pts.row(3) = Eigen::Vector3d(x_right, y_front_btm, z_btm).transpose();
    // blt
    matrix_8pts.row(4) = Eigen::Vector3d(x_left, y_back_top, z_top).transpose();
    // brt
    matrix_8pts.row(5) = Eigen::Vector3d(x_right, y_back_top, z_top).transpose();
    // blb
    matrix_8pts.row(6) = Eigen::Vector3d(x_left, y_back_btm, z_btm).transpose();
    // brb
    matrix_8pts.row(7) = Eigen::Vector3d(x_right, y_back_btm, z_btm).transpose();
}

typename Polyhedron::Halfedge_handle create_polyhedron_for_a_volume(
    const vector<double> &v, 
    Polyhedron & P){
    // based on the example from CGAL documentation: 
    // https://doc.cgal.org/latest/Polyhedron/index.html#PolyhedronExampleUsingEulerOperatorstoBuild

    // v: 0 x, 1 y, 2 z, 3 width, 4 height, 5 radius, 6 length of horizontal normal
    double nv = sqrt(1 - v[6] * v[6]);
    double z_top = v[2] + v[5] * nv + v[4] * v[6] / 2;
    double z_btm = v[2] + v[5] * nv - v[4] * v[6] / 2;
    double y_front_top = v[1] + v[5] * v[6] - v[4] / 2 * nv;
    double y_front_btm = v[1] + v[5] * v[6] + v[4] / 2 * nv;
    double y_back_top = v[1] - v[5] * v[6] + v[4] / 2 * nv;
    double y_back_btm = v[1] - v[5] * v[6] - v[4] / 2 * nv;
    double x_right = v[0] + v[3] / 2;
    double x_left = v[0] - v[3] / 2;

    CGAL_precondition(P.is_valid());
    typedef typename Polyhedron::Point_3         Point;
    typedef typename Polyhedron::Halfedge_handle Halfedge_handle;

    Point flt = Point(x_left, y_front_top, z_top);
    Point frt = Point(x_right, y_front_top, z_top);
    Point flb = Point(x_left, y_front_btm, z_btm);
    Point frb = Point(x_right, y_front_btm, z_btm);

    Point blt = Point(x_left, y_back_top, z_top);
    Point brt = Point(x_right, y_back_top, z_top);
    Point blb = Point(x_left, y_back_btm, z_btm);
    Point brb = Point(x_right, y_back_btm, z_btm);

    Halfedge_handle h = P.make_tetrahedron(brb, blt, blb, flb);
    Halfedge_handle g = h->next()->opposite()->next();             // Fig. (a)

    if (abs(y_front_top - y_back_top) > 0.01)
    {
        // cout << "y_front_top: " << y_front_top << ", y_back_top: " << y_back_top << endl;
        // cout << "normal case." << endl;
        
        P.split_edge(h->next());
        P.split_edge(g->next());
        P.split_edge(g);                                              // Fig. (b)
        h->next()->vertex()->point()     = brt;
        g->next()->vertex()->point()     = flt;
        g->opposite()->vertex()->point() = frb;                        // Fig. (c)
        Halfedge_handle f = P.split_facet(g->next(),
                                        g->next()->next()->next());    // Fig. (d)
        Halfedge_handle e = P.split_edge(f);
        e->vertex()->point() = frt;                                    // Fig. (e)
        P.split_facet( e, f->next()->next());                          // Fig. (f)
    }
    else
    {
        // cout << "y_front_top: " << y_front_top << ", y_back_top: " << y_back_top << endl;
        // cout << "pay attention to this specific case!" << endl;
        P.split_edge(h->next());
        P.split_edge(g); 
        h->next()->vertex()->point()     = brt;
        g->opposite()->vertex()->point() = frb;
        Halfedge_handle f = P.split_facet(g->prev(), g->next()->next());
    }
    CGAL_postcondition(P.is_valid());
    return h;
}


void gen_all_meshes(
    const vector<vector<double>> vol_params, 
    vector<Nef_Polyhedron> & vol_n_polyhedra)
{
    size_t vid = 0;
    for (auto v : vol_params)
    {
        if (v[3] > 0 && v[4] > 0 && v[5] > 0)
        {
            Polyhedron p;
            create_polyhedron_for_a_volume(v, p);
            Nef_Polyhedron n(p);
            vol_n_polyhedra.push_back(n);
            cout << "created " << vid << "th volume!" << endl;
            vid ++;
        }
    }
    cout << "*** created " << vol_params.size() << " volumes!" << endl;
}

size_t tree_union(vector<Nef_Polyhedron> vol_n_polyhedra, Nef_Polyhedron &out_n){
    vector<Nef_Polyhedron> out_n_polyhedra;
    bool to_union = true;
    size_t union_steps = 0;
    while(to_union){
        size_t num_to_union = vol_n_polyhedra.size();
        for (size_t i = 0; i < num_to_union; i=i+2){
            if (i == num_to_union - 1){
                out_n_polyhedra.push_back(vol_n_polyhedra[i]);
            }
            else{
                Nef_Polyhedron out;
                out = vol_n_polyhedra[i] + vol_n_polyhedra[i+1]; 
                out_n_polyhedra.push_back(out);
                union_steps ++;
            }
        }

        if (out_n_polyhedra.size() == 1){
            to_union = false;
            out_n = out_n_polyhedra[0];
        }
        else{
            vol_n_polyhedra.clear();
            vol_n_polyhedra = out_n_polyhedra;
            out_n_polyhedra.clear();
        }
    }
    return union_steps;
}

double computing_volume_of_nef_polyhedron(
    const Nef_Polyhedron& nef)
{
    Mesh mesh;
    try
    {
        CGAL::convert_nef_polyhedron_to_polygon_mesh(nef, mesh, true);
    }
    catch(const std::exception& e)
    {
        cout << e.what() << endl;
        vector<Point_3> points;
        vector<vector<size_t>> polygons;
        CGAL::convert_nef_polyhedron_to_polygon_soup(
            nef, points, polygons, true);
        // cout << "converted nef into points and polygons " << endl;
        PMP::repair_polygon_soup(points, polygons);
        // cout << "repaired polygon soup " << endl;
        PMP::orient_polygon_soup(points, polygons);
        // cout << "oriented polygon soup " << endl;
        PMP::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    }
        // cout << "converted polygon soup to polygon mesh " << endl;
    PMP::stitch_borders(mesh);
    bool bound_volume;
    try{
        bound_volume = PMP::does_bound_a_volume(mesh);
    }
    catch(const std::exception& e)
    {
        cerr << "Failed to compute the volume!" << endl;
        return 1e7;
    }
    if (bound_volume)
    {
        double volume = CGAL::to_double(PMP::volume(mesh));
        return  volume; // merge j to i
    }
    else
    {
        cerr << "Failed to compute the volume!" << endl;
        return 1e7;
    }
}

double compute_volume_from_param(
    const vector<double> param)
{
    double height = param[4] * param[6];
    double sinn = sqrt(1 - param[6] * param[6]);
    double width = param[3];
    double depth = (param[5] * param[6] + param[4]/2*sinn) * 2;
    double vol_vol = height * width * depth;

    double vol_fb_d = param[4] * sinn;
    double vol_cut = width * vol_fb_d * height;
    vol_vol -= vol_cut;

    return vol_vol;
}

bool is_possible_to_merge(
    const size_t i, const size_t j, const double buffer_dist,
    const Eigen::VectorXd& min_allx, const Eigen::VectorXd& max_allx,
    const Eigen::VectorXd& min_ally, const Eigen::VectorXd& max_ally,
    const Eigen::VectorXd& min_allz, const Eigen::VectorXd& max_allz)
{
    if (min_allx(i) > max_allx(j) + buffer_dist 
        || min_allx(j) > max_allx(i) + buffer_dist
        || min_ally(i) > max_ally(j) + buffer_dist
        || min_ally(j) > max_ally(i) + buffer_dist
        || min_allz(i) > max_allz(j) + buffer_dist
        || min_allz(j) > max_allz(i) + buffer_dist)

        return false;
    else
        return true;
}




pair<double, int> computing_merging_cost_of_one_pair(
    const Nef_Polyhedron& plh_i,
    const Nef_Polyhedron& plh_j,
    const Nef_Polyhedron& plh_nbh,
    const Nef_Polyhedron& plh_merge)
{
    Nef_Polyhedron nbh_diff, diff;
    double vol_nbh_diff, vol_diff, vol_i, vol_j;
    vol_i = computing_volume_of_nef_polyhedron(plh_i);
    vol_j = computing_volume_of_nef_polyhedron(plh_j);
    assert (vol_i > 0 && vol_j > 0);
    try
    {
        diff = plh_i - plh_j;

        if (diff == Nef_Polyhedron::EMPTY)
            vol_diff = 0;
        else
            vol_diff = computing_volume_of_nef_polyhedron(diff);
    }
    catch(const std::exception& e)
    {
        CGAL::draw(diff);
        vol_diff = 1e7;
    }
    
    try
    {
        nbh_diff = plh_merge - plh_nbh;

        if (nbh_diff == Nef_Polyhedron::EMPTY)
            vol_nbh_diff = 0;
            
        else
            vol_nbh_diff = computing_volume_of_nef_polyhedron(nbh_diff);
    }
    catch(const std::exception& e)
    {
        // CGAL::draw(new_vol);
        vol_nbh_diff = 1e7;
    }
    
    if (vol_diff < vol_nbh_diff)
        return make_pair(vol_diff, -1);
    else
        return make_pair((vol_i / vol_j) * vol_nbh_diff, 1);
}


void project_pts_to_update_volume(
    const Eigen::MatrixXd& pts,
    const vector<double>& param,
    vector<double>& updated_param)
{
    cout << "pts " << pts << endl;
    updated_param = param;
    // cout << "original param " ;
    // for (auto p : updated_param)
    //     cout << p << ", " ;
    // cout << endl;
    Eigen::Vector3d vol_center;
    vol_center << param[0], param[1], param[2];
    Eigen::MatrixXd pts_to_center_dist = pts;
    pts_to_center_dist.rowwise() -= vol_center.transpose();
    // the sign of y shouldn't affect the updating
    pts_to_center_dist.col(1) = pts_to_center_dist.col(1).cwiseAbs(); 
    // cout << "pts_to_center_dist: " << pts_to_center_dist << endl; 

    double sinn = sqrt(1 - param[6] * param[6]);
    double tann = sinn / param[6];
    Eigen::Vector3d y_dir;
    y_dir << 0, param[6], sinn;
    Eigen::Vector3d z_dir;
    z_dir << 0, -sinn, param[6];

    Eigen::VectorXd dist_y = pts_to_center_dist * y_dir;
    // Eigen::VectorXd proj_z = pts_to_center_dist * z_dir;
    // cout << "proj z: " << proj_z.transpose() << endl;
    // cout << "dist y: " << dist_y.transpose() << endl;

    // Update X
    double max_X_of_pts = pts.col(0).maxCoeff(), min_X_of_pts = pts.col(0).minCoeff();
    double max_X_of_vol = param[0] + param[3] / 2, min_X_of_vol = param[0] - param[3] / 2;
    double max_X = max_X_of_pts > max_X_of_vol ? max_X_of_pts : max_X_of_vol;
    double min_X = min_X_of_pts < min_X_of_vol ? min_X_of_pts : min_X_of_vol;
    updated_param[0] = (max_X + min_X) / 2;
    updated_param[3] = max_X - min_X;

    // Update zc
    double max_dist_y = dist_y.maxCoeff();
    double r = max_dist_y > updated_param[5] ? max_dist_y : updated_param[5];
    double zc = updated_param[2] + r / sinn;

    // update Z
    double max_Z_of_pts = pts.col(2).maxCoeff(), min_Z_of_pts = pts.col(2).minCoeff();
    double max_Z_of_vol = param[2] + updated_param[5]*sinn + param[4]/2 * param[6],
           min_Z_of_vol = param[2] + updated_param[5]*sinn - param[4]/2 * param[6];
    double max_Z = max_Z_of_pts > max_Z_of_vol ? max_Z_of_pts : max_Z_of_vol;
    double min_Z = min_Z_of_pts < min_Z_of_vol ? min_Z_of_pts : min_Z_of_vol;
    updated_param[4] = (max_Z - min_Z) / param[6];
    updated_param[5] = ((zc - (max_Z + min_Z)/2) * tann) / param[6];
    updated_param[2] = (max_Z + min_Z)/2 - ((zc - (max_Z + min_Z)/2) * tann) * tann;
    // cout << "max Z: " << max_Z << ", min Z: " << min_Z << endl;
    // cout << "((zc - (max_Z + min_Z)/2) * tann) * tann: " << ((zc - (max_Z + min_Z)/2) * tann) * tann;
    // cout << "updated param, zc: " << zc << endl;
    // cout << "updated param " ;
    // for (auto p : updated_param)
    //     cout << p << ", " ;
    // cout << endl;
}


double precomputing_merging_cost(
    const vector<vector<double>>& vol_params,
    const size_t num_box,
    Eigen::MatrixXd& merging_cost,
    Eigen::MatrixXi& merge_in_out,
    vector<Eigen::MatrixXd>& all_8pts_vol,
    Eigen::VectorXd& min_allx, Eigen::VectorXd& max_allx,
    Eigen::VectorXd& min_ally, Eigen::VectorXd& max_ally,
    Eigen::VectorXd& min_allz, Eigen::VectorXd& max_allz,
    vector<size_t>& pairs, double sigma = 1.)
{
    size_t num_vol = vol_params.size();
    assert (merging_cost.rows() == num_vol && merging_cost.cols() == num_vol+1);
    assert (merge_in_out.rows() == num_vol && merge_in_out.cols() == num_vol);
    
    vector<Nef_Polyhedron> vol_n_polyhedra;
    gen_all_meshes(vol_params, vol_n_polyhedra);
    Nef_Polyhedron merged_n;
    cout << "begin to union all input volumes ... " << endl;
    size_t union_steps = tree_union(vol_n_polyhedra, merged_n);
    double total_volume = computing_volume_of_nef_polyhedron(merged_n);
    cout << "end union with total volume: " << total_volume << endl; 
    // preparing
    Eigen::VectorXd all_vol_vol(num_vol);

    for (size_t i = 0; i < num_vol; i++)
    {
        vector<double> v = vol_params[i];
        all_vol_vol(i) = compute_volume_from_param(v);
        // all_vol_vol(i) = computing_volume_of_nef_polyhedron(vol_n_polyhedra[i]);
        get_8pts_of_a_volume_from_param(v, all_8pts_vol[i]);
        min_allx(i) = all_8pts_vol[i].col(0).minCoeff();
        max_allx(i) = all_8pts_vol[i].col(0).maxCoeff();
        min_ally(i) = all_8pts_vol[i].col(1).minCoeff();
        max_ally(i) = all_8pts_vol[i].col(1).maxCoeff();
        min_allz(i) = all_8pts_vol[i].col(2).minCoeff();
        max_allz(i) = all_8pts_vol[i].col(2).maxCoeff();
    }

    // merge two
    Eigen::MatrixXd possible_to_merge(num_vol, num_vol);
    possible_to_merge.setIdentity();
    merging_cost.setIdentity();
    merging_cost *= 1e7;
    merging_cost = 1e7 - merging_cost.array();
    merging_cost.col(num_vol) = all_vol_vol * 2; // cost of deleting the volume / merging the volume to NULL
    merge_in_out.setZero();
    // generate meaningful pairs: box & vol to box; box & vol to vol
    vector<pair<size_t, size_t>> Z1_pairs, Z2_pairs, Z3_pairs, Z4_pairs;
    size_t num_roof = num_vol - num_box;

    vector<Nef_Polyhedron> merged_for_each_row = vol_n_polyhedra;

    // make Z1 pairs
    for (size_t i = 0; i < num_box; i++)
        for (size_t j = i+1; j < num_box; j++)
            if (is_possible_to_merge(
                i, j, sigma, 
                min_allx, max_allx, 
                min_ally, max_ally, 
                min_allz, max_allz))
    {
        Z1_pairs.push_back(make_pair(i, j));
        possible_to_merge(i, j) = 1;
        possible_to_merge(j, i) = 1;
    }
    cout << "created " << Z1_pairs.size() << " Z1_pairs!" << endl;
    
    // make Z2 pairs
    for (size_t i = num_box; i < num_vol; i++)
        for (size_t j = 0; j < num_box; j++)
            if (is_possible_to_merge(
                i, j, sigma, 
                min_allx, max_allx, 
                min_ally, max_ally, 
                min_allz, max_allz))
    {
        Z2_pairs.push_back(make_pair(i, j));
        possible_to_merge(i, j) = 1;
    }
    cout << "created " << Z2_pairs.size() << " Z2_pairs!" << endl;
    // make Z3 pairs
    for (size_t i = 0; i < num_box; i++)
        for (size_t j = num_box; j < num_vol; j++)
            if (is_possible_to_merge(
                i, j, sigma, 
                min_allx, max_allx, 
                min_ally, max_ally, 
                min_allz, max_allz))
    {
        Z3_pairs.push_back(make_pair(i, j));
        possible_to_merge(i, j) = 1;

    }    
    for (size_t i = num_box; i < num_vol; i++)
        for (size_t j = i+1; j < num_vol; j++)
            if (is_possible_to_merge(
                i, j, sigma, 
                min_allx, max_allx, 
                min_ally, max_ally, 
                min_allz, max_allz))
    {
        Z3_pairs.push_back(make_pair(i, j));
        Z3_pairs.push_back(make_pair(j, i));
        possible_to_merge(i, j) = 1;
        possible_to_merge(j, i) = 1;
    }
    cout << "created " << Z3_pairs.size() << " Z3_pairs!" << endl;
    pairs.clear();
    pairs.resize(3);
    pairs[0] = Z1_pairs.size();
    pairs[1] = Z2_pairs.size();
    pairs[2] = Z3_pairs.size();

    // pre merge each row
    # pragma omp parallel for num_threads(16) 
    for (size_t i = 0; i < num_vol; i++)
        for (size_t j = 0; j < num_vol; j++)
    {
        if (possible_to_merge(i, j)>0)
            merged_for_each_row[i] += vol_n_polyhedra[j];
    }
    
    cout << "processing Z1_pairs " << endl;
    # pragma omp parallel for num_threads(16) 
    for (auto p : Z1_pairs)
    {
        vector<double> merged_v(7);
        auto start = chrono::high_resolution_clock::now();
        size_t i = p.first, j = p.second;
        // cout << "Z1 i: " << i << ", j: " << j ;
        double 
            min_x = min_allx(i) < min_allx(j) ? min_allx(i) : min_allx(j),
            max_x = max_allx(i) > max_allx(j) ? max_allx(i) : max_allx(j),
            min_y = min_ally(i) < min_ally(j) ? min_ally(i) : min_ally(j),
            max_y = max_ally(i) > max_ally(j) ? max_ally(i) : max_ally(j),
            min_z = min_allz(i) < min_allz(j) ? min_allz(i) : min_allz(j),
            max_z = max_allz(i) > max_allz(j) ? max_allz(i) : max_allz(j);
        merged_v[0] = (min_x + max_x) / 2;
        merged_v[1] = (min_y + max_y) / 2;
        merged_v[2] = (min_z + max_z) / 2;
        merged_v[3] = max_x - min_x;
        merged_v[4] = max_z - min_z;
        merged_v[5] = (max_y - min_y) / 2;
        merged_v[6] = 1.0;
        // double cost = computing_merging_cost_of_one_pair(merged_for_each_row[i], merged_v);
        Polyhedron merged_p;
        create_polyhedron_for_a_volume(merged_v, merged_p);
        Nef_Polyhedron new_vol(merged_p);
        pair<double, int> result;
        result = computing_merging_cost_of_one_pair(vol_n_polyhedra[i], vol_n_polyhedra[j], merged_for_each_row[i], new_vol);
        merging_cost(i, j) = result.first;
        merge_in_out(i, j) = result.second;
        result = computing_merging_cost_of_one_pair(vol_n_polyhedra[j], vol_n_polyhedra[i], merged_for_each_row[j], new_vol);
        merging_cost(j, i) = result.first;
        merge_in_out(j, i) = result.second;
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        // cout << " with duration: " << duration.count()/1000000.0 << " sec" << endl;
    }

    cout << "processing Z2_pairs " << endl;
    # pragma omp parallel for num_threads(16) 
    for (auto p : Z2_pairs)
    {
        vector<double> merged_v(7);
        auto start = chrono::high_resolution_clock::now();
        size_t i = p.first, j = p.second;
        double 
            min_x = min_allx(i) < min_allx(j) ? min_allx(i) : min_allx(j),
            max_x = max_allx(i) > max_allx(j) ? max_allx(i) : max_allx(j),
            min_y = min_ally(i) < min_ally(j) ? min_ally(i) : min_ally(j),
            max_y = max_ally(i) > max_ally(j) ? max_ally(i) : max_ally(j),
            min_z = min_allz(i) < min_allz(j) ? min_allz(i) : min_allz(j),
            max_z = max_allz(i) > max_allz(j) ? max_allz(i) : max_allz(j);
        merged_v[0] = (min_x + max_x) / 2;
        merged_v[1] = (min_y + max_y) / 2;
        merged_v[2] = (min_z + max_z) / 2;
        merged_v[3] = max_x - min_x;
        merged_v[4] = max_z - min_z;
        merged_v[5] = (max_y - min_y) / 2;
        merged_v[6] = 1.0;
        // double cost = computing_merging_cost_of_one_pair(merged_n, merged_v);
        Polyhedron merged_p;
        create_polyhedron_for_a_volume(merged_v, merged_p);
        Nef_Polyhedron new_vol(merged_p);
        pair<double, int> result = computing_merging_cost_of_one_pair(
                        vol_n_polyhedra[i], vol_n_polyhedra[j], merged_for_each_row[i], new_vol);
        merging_cost(i, j) = result.first;
        merge_in_out(i, j) = result.second;
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        // cout << "Z2 i: " << i << ", j: " << j 
        //      << " with duration: " << duration.count()/1000000.0 << " sec" << endl;
    }

    cout << "processing Z3_pairs " << endl;
    # pragma omp parallel for num_threads(16)
    for (auto p : Z3_pairs)
    { 
        vector<double> merged_v(7);
        auto start = chrono::high_resolution_clock::now();
        size_t i = p.first, j = p.second;
        cout << "Z3 i: " << i << ", j: " << j << endl;
        vector<double> vi = vol_params[i], 
                    vj = vol_params[j];
        double sinn = sqrt(1 - vj[6] * vj[6]),
            tann = sinn / vj[6],
            zc = vj[2] + vj[5] / sinn;
        
        Eigen::MatrixXd i_8pts = all_8pts_vol[i];
            
        Eigen::VectorXd proj_zony_of_i8pts = 
            i_8pts.col(2).array() + 
            (i_8pts.col(1).array() - vj[1]).abs() / tann;
        
        double max_proj_zony_of_i8pts = proj_zony_of_i8pts.maxCoeff();
        if (max_proj_zony_of_i8pts > zc)
            vj[5] = (max_proj_zony_of_i8pts - vj[2]) * sinn;
    
        project_pts_to_update_volume(i_8pts, vj, merged_v);   
        // double cost = computing_merging_cost_of_one_pair(merged_n, merged_v);
        Polyhedron merged_p;
        create_polyhedron_for_a_volume(merged_v, merged_p);
        Nef_Polyhedron new_vol(merged_p);
        pair<double, int> result = computing_merging_cost_of_one_pair(
                    vol_n_polyhedra[i], vol_n_polyhedra[j], merged_for_each_row[i], new_vol);
        merging_cost(i, j) = result.first;
        merge_in_out(i, j) = result.second;
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        // cout << "Z3 i: " << i << ", j: " << j 
        //      << " with duration: " << duration.count()/1000000.0 << " sec" << endl;
    }

    // for (size_t i = 0; i < num_vol-1; i++)
    // {
    //     for (size_t j = i+1; j < num_vol; j++)
    //     {
    //         if (merge_in_out(i, j) == -1 && merge_in_out(j, i) == 1)
    //         {
    //             merging_cost(j, i) = 1e7;
    //             merge_in_out(j, i) = 0;
    //         }
    //         else if(merge_in_out(j, i) == -1 && merge_in_out(i, j) == 1)
    //         {
    //             merging_cost(i, j) = 1e7;
    //             merge_in_out(i, j) = 0;
    //         }
    //     }
    // }

    return total_volume;
}


bool binary_optimization(
    const size_t num_vol,
    const Eigen::MatrixXd& merging_cost,
    const double lambda,
    // const double lambda_1, 
    // const double lambda_2,
    const double total_volume,
    Eigen::MatrixXd& is_mergedto)
{
    size_t num_merging_var = num_vol * num_vol / 2;
    assert (merging_cost.rows() == num_vol && merging_cost.cols() == num_vol+1); // ij: merge j to i
    GRBEnv env = GRBEnv();
    GRBModel model = GRBModel(env);
    vector<vector<GRBVar>> X;
    vector<GRBVar> COL_COUNT(num_vol);

    // Add variables
    for (size_t i = 0; i < num_vol; ++i)
    {
        vector<GRBVar> Xi(num_vol+1);
        for (size_t j = 0; j < num_vol + 1; ++j)
        {
            if (i == j)
                continue;
            // whether merge i to j
            Xi[j] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, 
                    "merge " + to_string(i)+" to "+ to_string(j));
        } 
        Xi[num_vol] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, 
                    "merge " + to_string(i) + " to NULL"); // whether assign to NULL
        X.push_back(Xi);
        COL_COUNT[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "count col - " + to_string(i));
    }
    model.update();

    // Add constraints
    for (size_t i = 0; i < num_vol; ++i)
    {
        GRBLinExpr exp_row, exp_col;
        for (size_t j = 0; j < num_vol; ++j)
        {
            if (i == j)
                continue;
            exp_row += X[i][j];
            exp_col += X[j][i];
        }  
        exp_row += X[i][num_vol];
        model.addConstr(COL_COUNT[i] - (exp_col + 1 - exp_row)/num_vol >=0); 
        exp_col /= num_vol;

        model.addConstr(exp_col + exp_row <= 1.0); // only merge to the one that isn't be merged to the other
    }

    // Set objective
    GRBLinExpr obj;
    // obj = lambda_1;
    obj = 0;
    for (size_t i = 0; i < num_vol; ++i)
    {
        for (size_t j = 0; j < num_vol+1; ++j)
        {
            if (i == j)
                continue;
            obj += X[i][j] * (merging_cost(i,j) / total_volume);
            // obj += X[i][j] * (merging_cost(i,j) / 1000);
            // obj -= lambda_1 * X[i][j] / num_merging_var;
        }
        obj += lambda * COL_COUNT[i] / num_vol;
    }
    model.setObjective(obj, GRB_MINIMIZE);
    model.optimize();
    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL)
    {
        double obj_value = model.get(GRB_DoubleAttr_ObjVal);
        // 把两部分的数值print出来看看
        assert (is_mergedto.rows() == num_vol && is_mergedto.cols() == num_vol+1);
        is_mergedto.setZero();
        for (size_t i = 0; i < num_vol; i++)
        {   
            cout << "checking optimization result of i: " << i << endl;
            size_t row_v = 0, col_v = 0;

            for (size_t j = 0; j < num_vol+1; j++)
            {
                if (i == j)
                    continue;
                is_mergedto(i,j) = bool(X[i][j].get(GRB_DoubleAttr_X));
                row_v += size_t(X[i][j].get(GRB_DoubleAttr_X));
                if (j < num_vol)
                    col_v += size_t(X[j][i].get(GRB_DoubleAttr_X));
            }
            cout << "row_v: " << row_v << ", col_v: " << col_v << endl;
            if (is_mergedto.row(i).sum() == 0)
                is_mergedto(i,i) = 1;
        }
        cout << "merging cost: " << merging_cost << endl;
        cout << "is merged to: " << is_mergedto << endl;
            
        return true;
    } 
    else
    {
        cerr << "optimziation status: " << status << endl;
        return false;
    }
}


void parsing_merging_result(
    const size_t num_vol,
    const vector<vector<double>>& input_vol_params,
    const Eigen::MatrixXd& is_mergedto,
    const Eigen::MatrixXi& merge_in_out,
    const vector<Eigen::MatrixXd>& all_8pts_vol,
    const Eigen::VectorXd& min_allx, const Eigen::VectorXd& max_allx, 
    const Eigen::VectorXd& min_ally, const Eigen::VectorXd& max_ally, 
    const Eigen::VectorXd& min_allz, const Eigen::VectorXd& max_allz, 
    vector<vector<double>>& merged_vol_param,
    vector<Eigen::MatrixXd>& merged_vol_8pts)
{
    // processing each column
    for (size_t i = 0; i < num_vol; i++)
    {
        if (is_mergedto(i, i)==1)
        {
            vector<double> ovi = input_vol_params[i];
            Eigen::MatrixXd ovi_8pts = all_8pts_vol[i]; 
            vector<double> merged_v(7);
            size_t num_merged_in = 0;
            for (size_t j = 0; j < num_vol; j++)
            {
                if (is_mergedto(j, i) == 1 && merge_in_out(j, i) == 1)
                    num_merged_in ++;
            }
                
            if (num_merged_in > 0)
            { 
                cout << "merging to i: " << i << endl;
                // get all points that would be merged into Volume i
                
                
                size_t num_pts = 8 * num_merged_in;
                Eigen::MatrixXd all_8pts (num_pts, 3);
                size_t jid = 0;
                for (size_t j = 0 ; j < num_vol; j++)
                {
                    if (i == j)
                        continue;
                    if(is_mergedto(j, i)==1 && merge_in_out(j, i) == 1)
                    {
                        all_8pts.block<8,3>(jid, 0) = all_8pts_vol[j];
                        jid += 8;
                    }
                }
                
                // compute the new parameter of Volume i
                // ovi refer to original_vol_i 
                double min_all_x = all_8pts.col(0).minCoeff(),
                    max_all_x = all_8pts.col(0).maxCoeff(),
                    min_all_y = all_8pts.col(1).minCoeff(),
                    max_all_y = all_8pts.col(1).maxCoeff(),
                    min_all_z = all_8pts.col(2).minCoeff(),
                    max_all_z = all_8pts.col(2).maxCoeff();
                
                cout << "min_all_x: " << min_all_x << ", max_all_x: " << max_all_x << endl
                     << "min_all_y: " << min_all_y << ", max_all_y: " << max_all_y << endl
                     << "min_all_z: " << min_all_z << ", max_all_z: " << max_all_z << endl
                     << endl;
                cout << "min vol x: " << min_allx(i) << ", max vol x: " << max_allx(i) << endl
                     << "min vol y: " << min_ally(i) << ", max vol y: " << max_ally(i) << endl
                     << "min vol z: " << min_allz(i) << ", max vol z: " << max_allz(i) << endl
                     << endl;


                if (ovi[6] == 1) // i.e., box
                {
                    double min_x = min_allx(i) < min_all_x ? min_allx(i) : min_all_x,
                       min_y = min_ally(i) < min_all_y ? min_ally(i) : min_all_y,
                       min_z = min_allz(i) < min_all_z ? min_allz(i) : min_all_z,
                       max_x = max_allx(i) > max_all_x ? max_allx(i) : max_all_x,
                       max_y = max_ally(i) > max_all_y ? max_ally(i) : max_all_y,
                       max_z = max_allz(i) > max_all_z ? max_allz(i) : max_all_z;
                    
                    cout << "min x: " << min_x << ", max x:" << max_x 
                         << "min y: " << min_y << ", max y: " << max_y
                         << "min z: " << min_z << ", max z: " << max_z
                         << endl;

                    merged_v[0] = (min_x + max_x) / 2;
                    merged_v[1] = (min_y + max_y) / 2;
                    merged_v[2] = (min_z + max_z) / 2;
                    merged_v[3] = max_x - min_x;
                    merged_v[4] = max_z - min_z;
                    merged_v[5] = (max_y - min_y) / 2;
                    merged_v[6] = 1.0;
                    
                }
                else // roof, i.e., original_vol_i[6] < 1
                {
                    double sinn = sqrt(1 - ovi[6] * ovi[6]),
                           tann = sinn / ovi[6],
                           zc = ovi[6] + ovi[5] / sinn;
                    Eigen::VectorXd proj_zony_of_allpts = 
                        all_8pts.col(2).array()
                        + (all_8pts.col(1).array() - ovi[1]).abs()/tann;
                    double max_proj_zony_of_allpts = proj_zony_of_allpts.maxCoeff();

                    if (max_proj_zony_of_allpts > zc)
                        ovi[5] = (max_proj_zony_of_allpts - ovi[2]) * sinn;

                    project_pts_to_update_volume(all_8pts, ovi, merged_v);
                }
            }
            else
                merged_v = input_vol_params[i];
            
            merged_vol_param.push_back(merged_v);
            Eigen::MatrixXd vol_8pts(8, 3);
            get_8pts_of_a_volume_from_param(merged_v, vol_8pts);
            merged_vol_8pts.push_back(vol_8pts);
        }
    }
}


void save_vol_8pts_as_obj(
    const vector<Eigen::MatrixXd>& merged_8pts,
    const string& obj_path)
{
    vector<Point_3> points;
    vector<vector<size_t>> polygons;

    size_t num_pts = 0;
    for (auto pts : merged_8pts)
    {
        // add points
        // flt
        points.push_back(Point_3(pts(0,0), pts(0,1), pts(0,2)));
        // frt
        points.push_back(Point_3(pts(1,0), pts(1,1), pts(1,2)));
        // flb
        points.push_back(Point_3(pts(2,0), pts(2,1), pts(2,2)));
        // frb
        points.push_back(Point_3(pts(3,0), pts(3,1), pts(3,2)));
        // blt
        points.push_back(Point_3(pts(4,0), pts(4,1), pts(4,2)));
        // brt
        points.push_back(Point_3(pts(5,0), pts(5,1), pts(5,2)));
        // blb
        points.push_back(Point_3(pts(6,0), pts(6,1), pts(6,2)));
        // brb
        points.push_back(Point_3(pts(7,0), pts(7,1), pts(7,2)));

        // add faces
        // top: FLT -> BLT -> BRT -> FRT 
        polygons.push_back(vector<size_t>{
            num_pts, num_pts+4, num_pts+5, num_pts+1});
        // bottom: FLB ->FRB -> BRB -> BLB
        polygons.push_back(vector<size_t>{
            num_pts+2, num_pts+3, num_pts+7, num_pts+6});
        // front: FLT -> FRT -> FRB -> FLB
        polygons.push_back(vector<size_t>{
            num_pts, num_pts+1, num_pts+3, num_pts+2});
        // back: BLT -> BLB -> BRB -> BRT
        polygons.push_back(vector<size_t>{
            num_pts+4, num_pts+6, num_pts+7, num_pts+5});
        // left: FLT -> FLB -> BLB -> BLT
        polygons.push_back(vector<size_t>{
            num_pts, num_pts+2, num_pts+6, num_pts+4});
        // right: FRT -> BRT -> BRB -> FRB
        polygons.push_back(vector<size_t>{
            num_pts+1, num_pts+5, num_pts+7, num_pts+3});
        num_pts += 8;
    }
    CGAL::IO::write_OBJ(obj_path, points, polygons);
}

void save_vol_param(
    const vector<vector<double>>& params,
    const string& param_path)
{
    ofstream params_out(param_path);
    for (auto param : params)
        params_out << param[0] << " "
                   << param[1] << " "
                   << param[2] << " "
                   << param[3] << " "
                   << param[4] << " "
                   << param[5] << " "
                   << param[6] << endl;
}