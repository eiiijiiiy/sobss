#include <iostream>
#include <fstream>
#include <sys/time.h>

#include <nlopt.hpp>
#include <set>
#include <vector>
#include <iomanip>
#include <random>
#include <iterator>
#include <algorithm>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmath.hpp>
#include <filesystem>

#include "src/util.h"
#include "src/Solvernlopt.hpp"



// for static obj and var in the nlopt
std::vector<double> Solvernlopt::frequency_ = std::vector<double>();
double Solvernlopt::interval_ = 0.0;
int Solvernlopt::half_PI_bin_num_ = 0;


enum EnumNormalGroup {horizontal=0, non_horizontal=1, other=2}; // normal group


inline bool exists (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

// group vertical and horizontal normals
pair<shared_ptr<PointCloud>, shared_ptr<PointCloud>> ClusterPointNormals(
            PointCloud &pcd, 
            std::vector<EnumNormalGroup> &labels,
            double h_z_thresh, 
            double v_z_thresh){ 
    // WITH SYMMETRY
    double orientation= 0.0;
    bool with_color = true;
    if (pcd.colors_.size() == 0) {
        with_color = false;
        cout << "* input point cloud without colors." << endl;
    }
    
    PointCloud non_horizontal_pcd = open3d::geometry::PointCloud();
    PointCloud horizontal_pcd = open3d::geometry::PointCloud();

    for (size_t i = 0; i < pcd.points_.size(); i++) {
        if (std::abs(pcd.normals_[i][2]) >= h_z_thresh) {
            labels[i] = EnumNormalGroup::horizontal;
            horizontal_pcd.points_.push_back(pcd.points_[i]);
            horizontal_pcd.normals_.push_back(pcd.normals_[i]);
            
            if (with_color)
                horizontal_pcd.colors_.push_back(pcd.colors_[i]);
        }
        else if ((std::abs(pcd.normals_[i][2]) <= v_z_thresh) ||
                    (pcd.normals_[i][2] > 0)){
            labels[i] = EnumNormalGroup::non_horizontal;
            non_horizontal_pcd.points_.push_back(pcd.points_[i]);
            non_horizontal_pcd.normals_.push_back(pcd.normals_[i]);

            if (with_color)
                non_horizontal_pcd.colors_.push_back(pcd.colors_[i]);
        }
    }    
    return pair<shared_ptr<PointCloud>, shared_ptr<PointCloud>>(
        make_shared<PointCloud>(non_horizontal_pcd), make_shared<PointCloud>(horizontal_pcd));
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
         << endl;
    double sum_x = 0;
    double sum_y = 0;

    std::vector<double> normal_radian_frequency(half_PI_bin_num, 0.0);
    double interval = M_PI / (half_PI_bin_num * 2);

    for(size_t i = 0; i < num_point; i++) {
        Eigen::Vector2f horizontal_normal = Eigen::Vector2f(vertical_pcd.normals_[i][0], 
                                    vertical_pcd.normals_[i][1]).normalized();
        double radian = std::atan2(horizontal_normal[1], horizontal_normal[0]);
        int bin_id = std::floor(radian / interval);
        bin_id = bin_id - std::floor( (double)bin_id / half_PI_bin_num) * half_PI_bin_num;
        normal_radian_frequency[bin_id] += (double)1/num_point;
    }
    cout << "* normal_radian_frequency computed." << endl;
    // DIRECT (NLOpt) search 
    Solvernlopt nloptsolver(normal_radian_frequency, interval, half_PI_bin_num);
    return nloptsolver.solve(nlopt::algorithm::GN_DIRECT, max_iter);
}


/// \brief weighted octree sampling (rewritten version of odas)
std::pair<shared_ptr<PointCloud>, std::vector<float>> 
    weighted_downsample(const PointCloud &pcd, const size_t & max_depth){

    Octree oct = Octree(max_depth);
    oct.ConvertFromPointCloud(pcd);
    size_t pt_num = pcd.points_.size();

    // traverse function to count weights of leaf nodes

    //TODO: pcd uncaptured problem
    PointCloud sampled_pcd = open3d::geometry::PointCloud();
    std::vector<float> weights;
    auto f_sample = 
        [pcd, &sampled_pcd, &weights](
        const std::shared_ptr<OctreeNode>& src_node,
        const std::shared_ptr<OctreeNodeInfo>& src_node_info)
        -> bool {
            if(auto src_leaf_node = 
                    std::dynamic_pointer_cast<OctreePointColorLeafNode>(src_node)){
                // count points
                size_t pt_count = src_leaf_node->indices_.size();
                float weight = float(pt_count) / pcd.points_.size();
                weights.push_back(weight);
                // get points' center
                Eigen::Vector3d sum_pt(0, 0, 0); // should be vector3d since the points_ member in PointCloud is in double
                for (size_t i = 0; i < pt_count; i++){
                    sum_pt += pcd.points_[src_leaf_node->indices_[i]];
                }
                sampled_pcd.points_.push_back(sum_pt / pt_count);
                
            }
            return false;
    };
    oct.Traverse(f_sample);
    
    return pair<shared_ptr<PointCloud>, vector<float>>(
            make_shared<PointCloud>(sampled_pcd), weights);
}


void create_3D_model_of_boxes(
    const vector<vector<double>> &locations,
    const vector<vector<double>> &dimensions,
    const vector<vector<double>> &directions,
    string output_path){

    std::ofstream obj_out(output_path);
    vector<Eigen::Vector2d> coords_2d(locations.size() * 4);

    obj_out << "Ka 1.000000 1.000000 1.000000" << endl;
    obj_out << "Kd 1.000000 1.000000 1.000000" << endl;
    obj_out << "Ks 0.000000 0.000000 0.000000" << endl;
    obj_out << "Tr 1.000000" << endl;
    obj_out << "illum 1" << endl;
    obj_out << "Ns 0.000000 1" << endl;
    

    for (size_t i = 0 ; i < locations.size(); i++) {
        obj_out << "o instance: " << i << "n: " << "(" << to_string(directions[i][0]) 
                << ", " << to_string(directions[i][1]) << ")" << endl;
        double x, y, _x, _y;
        vector<double> vec1(2), vec2(2);
        x = locations[i][0];
        y = locations[i][1];
        _x = dimensions[i][0];
        _y = dimensions[i][1];
        vec1 = directions[i];
        vec2[0] = -vec1[1];
        vec2[1] = vec1[0];

        size_t bi = i * 4; // the begin index of 2D vertices
        size_t bi_ = i * 8 + 1; // the begin index of 3D vertices

        // compute 4 2D coordinates
        // index of faces: rut 0 rlt 1 lut 2 llt 3 rub 4 rlb 5 lub 6 llb 7
        // right upper
        coords_2d[bi][0] = x + (_x/2) * vec1[0] + (_y/2) * vec2[0];
        coords_2d[bi][1] = y + (_x/2) * vec1[1] + (_y/2) * vec2[1];
        // coords_2d[bi][0] = x + (_x / 2) * cos(theta) - (_y / 2) * sin(theta);
        // coords_2d[bi][1] = y + (_x / 2) * sin(theta) + (_y / 2) * cos(theta);
        // right lower 
        coords_2d[bi + 1][0] = x + (_x/2) * vec1[0] - (_y/2) * vec2[0];
        coords_2d[bi + 1][1] = y + (_x/2) * vec1[1] - (_y/2) * vec2[1];
        // coords_2d[bi + 1][0] = x + (_x / 2) * cos(theta) - (-_y / 2) * sin(theta);
        // coords_2d[bi + 1][1] = y + (_x / 2) * sin(theta) + (-_y / 2) * cos(theta);
        // left upper
        coords_2d[bi + 2][0] = x - (_x/2) * vec1[0] + (_y/2) * vec2[0];
        coords_2d[bi + 2][1] = y - (_x/2) * vec1[1] + (_y/2) * vec2[1];
        // coords_2d[bi + 2][0] = x + (-_x / 2) * cos(theta) - (_y / 2) * sin(theta);
        // coords_2d[bi + 2][1] = y + (-_x / 2) * sin(theta) + (_y / 2) * cos(theta);
        // left lower
        coords_2d[bi + 3][0] = x - (_x/2) * vec1[0] - (_y/2) * vec2[0];
        coords_2d[bi + 3][1] = y - (_x/2) * vec1[1] - (_y/2) * vec2[1];
        // coords_2d[bi + 3][0] = x + (-_x / 2) * cos(theta) - (-_y / 2) * sin(theta);
        // coords_2d[bi + 3][1] = y + (-_x / 2) * sin(theta) + (-_y / 2) * cos(theta);
        
        // top: rut -> lut -> llt -> rlt
        obj_out << "f " << bi_ << " " << bi_ + 2 << " " << bi_ + 3 << " " << bi_ + 1 << endl;
        // bottom: rlb -> llb -> lub -> rub
        obj_out << "f " << bi_ + 5 << " " << bi_ + 7 << " " << bi_ + 6 << " " << bi_ + 5 << endl;
        // left: llt -> lut -> lub -> llb
        obj_out << "f " << bi_ + 3 << " " << bi_ + 2 << " " << bi_ + 6 << " " << bi_ + 7 << endl;
        // right: rlb -> rub -> rut -> rlt
        obj_out << "f " << bi_ + 5 << " " << bi_ + 4 << " " << bi_ << " " << bi_ + 1 << endl;
        // front: rlt -> llt -> llb -> rlb
        obj_out << "f " << bi_ + 1 << " " << bi_ + 3 << " " << bi_ + 7 << " " << bi_ + 5 << endl;
        // back: rub -> lub -> lut -> rut
        obj_out << "f " << bi_ + 4 << " " << bi_ + 6 << " " << bi_ + 2 << " " << bi_ << endl;
    }

    for (size_t i = 0; i < locations.size(); i++){
        double z = locations[i][2];
        double half_z =  dimensions[i][2] / 2;

        obj_out << "v " << coords_2d[i*4][0] << " " << coords_2d[i*4][1] << " " << z + half_z << endl;
        obj_out << "v " << coords_2d[i*4+1][0] << " " << coords_2d[i*4+1][1] << " " << z + half_z << endl;
        obj_out << "v " << coords_2d[i*4+2][0] << " " << coords_2d[i*4+2][1] << " " << z + half_z << endl;
        obj_out << "v " << coords_2d[i*4+3][0] << " " << coords_2d[i*4+3][1] << " " << z + half_z << endl;

        obj_out << "v " << coords_2d[i*4][0] << " " << coords_2d[i*4][1] << " " << z - half_z << endl;
        obj_out << "v " << coords_2d[i*4+1][0] << " " << coords_2d[i*4+1][1] << " " << z - half_z << endl;
        obj_out << "v " << coords_2d[i*4+2][0] << " " << coords_2d[i*4+2][1] << " " << z - half_z << endl;
        obj_out << "v " << coords_2d[i*4+3][0] << " " << coords_2d[i*4+3][1] << " " << z - half_z << endl;
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
        best_dist = 1e100;
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
    string normal_group_prefix, 
    string output_voted_param_path, 
    string output_voted_center_pcd_path,
    vector<double> &process_t,
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
 
    gettimeofday(&t_start, NULL);

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
    gettimeofday(&t_end, NULL);
    sec = t_end.tv_sec - t_start.tv_sec;
    usec = t_end.tv_usec - t_start.tv_usec;
    timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    process_t.push_back(timecost);

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

    gettimeofday(&t_end, NULL);
    sec = t_end.tv_sec - t_start.tv_sec;
    usec = t_end.tv_usec - t_start.tv_usec;
    timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    process_t.push_back(timecost);

    /*****************SAVE THE PRIMARY NORMAL GROUP *******************/
    // <--speed test, do not count the time for saving results
    string positive_pcd_group_path = normal_group_prefix + "_p.ply";
    string negative_pcd_group_path = normal_group_prefix + "_n.ply";
    open3d::io::WritePointCloud(positive_pcd_group_path, ppcd);
    open3d::io::WritePointCloud(negative_pcd_group_path, npcd);
    // speed test, do not count the time for saving results -->
    /******************************************************************/
    gettimeofday(&t_start, NULL);
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
    gettimeofday(&t_end, NULL);
    sec = t_end.tv_sec - t_start.tv_sec;
    usec = t_end.tv_usec - t_start.tv_usec;
    timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
    process_t.push_back(timecost);

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


int main(int argc, char * argv[]){

    rj::Document config_doc;
    if (!read_config(argv[1], config_doc))
        return EXIT_FAILURE;

    string input_pcd_dir, save_group_root_dir, time_log_dir,
        voted_params_dir, voted_center_pcd_dir, output_axis_aligned_dir, time_log_path;
    double horizontal_normal_z_thresh, vertical_normal_z_thresh,
        voxel_size, normal_pair_thresh;
    int orientation_max_iter, half_PI_bin_num, sample_num;
    bool save_axis_aligned;

    if (config_doc.HasMember("input_pcd_dir") && config_doc["input_pcd_dir"].IsString())
        input_pcd_dir = config_doc["input_pcd_dir"].GetString();
    
    if (config_doc.HasMember("save_group_root_dir") && config_doc["save_group_root_dir"].IsString())
        save_group_root_dir = config_doc["save_group_root_dir"].GetString();
    
    if (config_doc.HasMember("time_log_dir") && config_doc["time_log_dir"].IsString())
        time_log_dir = config_doc["time_log_dir"].GetString();
    
    if (config_doc.HasMember("voted_params_dir") && config_doc["voted_params_dir"].IsString())
        voted_params_dir = config_doc["voted_params_dir"].GetString();
    
    if (config_doc.HasMember("voted_center_pcd_dir") && config_doc["voted_center_pcd_dir"].IsString())
        voted_center_pcd_dir = config_doc["voted_center_pcd_dir"].GetString();
    
    if (config_doc.HasMember("output_axis_aligned_dir") && config_doc["output_axis_aligned_dir"].IsString())
        output_axis_aligned_dir = config_doc["output_axis_aligned_dir"].GetString();
    
    if (config_doc.HasMember("horizontal_normal_z_thresh") && config_doc["horizontal_normal_z_thresh"].IsDouble())
        horizontal_normal_z_thresh = config_doc["horizontal_normal_z_thresh"].GetDouble();

    if (config_doc.HasMember("vertical_normal_z_thresh") && config_doc["vertical_normal_z_thresh"].IsDouble())
        vertical_normal_z_thresh = config_doc["vertical_normal_z_thresh"].GetDouble();
    
    if (config_doc.HasMember("voxel_size") && config_doc["voxel_size"].IsDouble())
        voxel_size = config_doc["voxel_size"].GetDouble();
    
    if (config_doc.HasMember("normal_pair_thresh") && config_doc["normal_pair_thresh"].IsDouble())
        normal_pair_thresh = config_doc["normal_pair_thresh"].GetDouble();
    
    if (config_doc.HasMember("orientation_max_iter") && config_doc["orientation_max_iter"].IsInt())
        orientation_max_iter = config_doc["orientation_max_iter"].GetInt();
    
    if (config_doc.HasMember("half_PI_bin_num") && config_doc["half_PI_bin_num"].IsInt())
        half_PI_bin_num = config_doc["half_PI_bin_num"].GetInt();
    
    if (config_doc.HasMember("sample_num") && config_doc["sample_num"].IsInt())
        sample_num = config_doc["sample_num"].GetInt();

    if (config_doc.HasMember("save_axis_aligned") && config_doc["save_axis_aligned"].IsBool())
        save_axis_aligned = config_doc["save_axis_aligned"].GetBool();

    cout << "configure parameters: "
         << "\t orientation_max_iter: " << orientation_max_iter << endl
         << "\t half_PI_bin_num: " << half_PI_bin_num << endl
         << "\t horizontal_normal_z_thresh: " << horizontal_normal_z_thresh << endl
         << "\t vertical_normal_z_thresh: " << vertical_normal_z_thresh << endl
         << "\t voxel_size: " << voxel_size << endl
         << "\t normal_pair_thresh: " << normal_pair_thresh << endl;
    time_log_path = time_log_dir + "vs_" + to_string(voxel_size) + ".txt";
    cout << "configure directories and file paths:"
        << "\t input_pcd_dir: " << input_pcd_dir << endl
        << "\t sample_num: " << sample_num << endl
        << "\t save_group_root_dir: " << save_group_root_dir << endl
        << "\t time_log_dir: " << time_log_dir << endl
        << "\t voted_params_dir: " << voted_params_dir << endl
        << "\t voted_center_pcd_dir: " << voted_center_pcd_dir << endl
        << "\t time_log_path: " << time_log_path << endl;

    ofstream time_log(time_log_path);
    vector<double> process_t;
    struct timeval t_start, t_end;
    long sec, usec;
    double timecost;
    // for (const auto & entry : fs::directory_iterator(input_pcd_dir)){
    for (int i = 1; i <= sample_num; i++){
        process_t.clear();
        string input_pcd_path = input_pcd_dir + to_string(i) + ".ply";
        cout << "* processing " << input_pcd_path << endl;

        std::shared_ptr<PointCloud> input_pcd; 
        input_pcd = open3d::io::CreatePointCloudFromFile(input_pcd_path);
        cout << "* loaded " << input_pcd_path << endl;
        if (input_pcd->points_.size() == 0) {
            cerr << "no points in " << input_pcd_path << endl;
            return 1;
        }
        // input should with Normals compuated by CloudCompare NOT open3d
        if (!input_pcd->HasNormals())
            cerr << "no normals in " << input_pcd_path << endl;
        
        Eigen::Vector3d input_pcd_center = input_pcd->GetCenter();
        input_pcd->Translate(-input_pcd_center);
        string data_name = to_string(i);

        std::shared_ptr<PointCloud> voting_pcd;
        std::vector<EnumNormalGroup> normal_labels(input_pcd->points_.size(), EnumNormalGroup::other);
        std::pair<std::shared_ptr<PointCloud>, std::shared_ptr<PointCloud>> result = ClusterPointNormals(
            *input_pcd, normal_labels, horizontal_normal_z_thresh, vertical_normal_z_thresh);
        std::shared_ptr<PointCloud> non_horizontal_pcd = std::get<0>(result);

        /****************** 0.1 AXIS ALIGN *******************/
        gettimeofday(&t_start, NULL);
        double orientation = EstimateHorizontalOrientation(*non_horizontal_pcd, half_PI_bin_num, orientation_max_iter);
        gettimeofday(&t_end, NULL);
        sec = t_end.tv_sec - t_start.tv_sec;
        usec = t_end.tv_usec - t_start.tv_usec;
        timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
        process_t.push_back(timecost);

        Eigen::AngleAxisd angle_axis(-orientation, Eigen::Vector3d(0,0,1));
        Eigen::Matrix3d rot_matrix = angle_axis.toRotationMatrix();
        Eigen::Vector3d rot_center = input_pcd->GetCenter();
        input_pcd->Rotate(rot_matrix, rot_center);
        non_horizontal_pcd->Rotate(rot_matrix, rot_center);
        cout << "rotated to align the axis. " << endl;

        /****************** SELECT PRIMARY NORMAL ******************/
        size_t num_pts = non_horizontal_pcd->points_.size();
        // group normals
        vector<Eigen::Vector2d> normals_2d(num_pts, Eigen::Vector2d(0,0));
        for(size_t i = 0; i < num_pts; i++){
            normals_2d[i](0) = abs(non_horizontal_pcd->normals_[i][0]);
            normals_2d[i](1) = abs(non_horizontal_pcd->normals_[i][1]);
            normals_2d[i].normalized();
        }
        vector<Eigen::Vector2d> ng = {Eigen::Vector2d(0, 1), Eigen::Vector2d(1, 0)};
        size_t num_ng = ng.size();
        vector<size_t> ng_id(num_pts);
        vector<vector<size_t>> ng_pt_idx(num_ng);

        gettimeofday(&t_start, NULL);
        group_normals(normals_2d, ng, ng_id, ng_pt_idx);
        gettimeofday(&t_end, NULL);
        sec = t_end.tv_sec - t_start.tv_sec;
        usec = t_end.tv_usec - t_start.tv_usec;
        timecost = (double(sec) + double(usec)/1000000.0) + 0.5/1000;
        process_t.push_back(timecost);

        size_t selected_normal_id = (ng_pt_idx[0].size() > ng_pt_idx[1].size()) ? 0:1 ;
        cout << "selected primary normal group " << ng_pt_idx[selected_normal_id].size() << "/" << num_pts
             << "(" << ng[selected_normal_id][0] << ", " << ng[selected_normal_id][1] << ")" << endl;
        
        if (selected_normal_id>0){
            Eigen::AngleAxisd angle_axis(M_PI/2, Eigen::Vector3d(0,0,1));
            Eigen::Matrix3d rot_matrix = angle_axis.toRotationMatrix();
            Eigen::Vector3d rot_center = non_horizontal_pcd->GetCenter();
            input_pcd->Rotate(rot_matrix, rot_center);
            non_horizontal_pcd->Rotate(rot_matrix, rot_center);
            cout << "* rotated the align the primary normal with (1, 0, 0)" << endl;
        }

       
        if (save_axis_aligned) {
            
            string output_axis_aligned_path = output_axis_aligned_dir + data_name + ".ply";
            open3d::io::WritePointCloud(output_axis_aligned_path, *input_pcd);
            cout << "saved axis-alinged point cloud to " << output_axis_aligned_path << endl;
        }

        /****************** 0.2 DOWN SAMPLE - ESTIMATE NORMAL******************/
        voting_pcd = non_horizontal_pcd;
        /****************** VOTE ******************/
        string save_group_prefix = save_group_root_dir + data_name;
        string output_voted_params_path = voted_params_dir + data_name + ".txt";
        string output_voted_center_pcd_path = voted_center_pcd_dir + data_name + ".ply";
        
        fast_vote(
            *voting_pcd,
            voxel_size,
            ng_pt_idx[selected_normal_id],
            save_group_prefix, 
            output_voted_params_path, output_voted_center_pcd_path,
            process_t,
            normal_pair_thresh, vertical_normal_z_thresh);
        time_log << process_t[0] << " " << process_t[1] << " " 
                << process_t[2] << " " << process_t[3] << " " << process_t[4] << " "<< endl;
    }

    return 0;
}