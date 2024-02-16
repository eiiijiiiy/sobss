#include "util.hpp"

int collect_bss_atoms(
    const string pcd_path, 
    const string working_folder) 
{
    // rj::Document config_doc;
    string conf_path = working_folder + "/config.json";
    if (!read_config(conf_path.c_str(), config_doc))
        return EXIT_FAILURE;
    
    double voxel_size;
    if (conf_doc.HasMember("voxel_size") && conf_doc["voxel_size"].IsDouble())
        voxel_size = config_doc["voxel_size"].GetDouble();
    else
    {
        cerr << "voxel_size is not set in " << conf_path << endl;
        cout << "set voxel_size as 2 m by default. \n";
        voxel_size = 2;
    }

    // 1. read point cloud in
    shared_ptr<PointCloud> in_pcd; 
    in_pcd = open3d::io::CreatePointCloudFromFile(pcd_path);
    cout << "* loaded " << pcd_path << endl;
    if (in_pcd->points_.size() == 0) {
        cerr << "no points in " << pcd_path << endl;
        return EXIT_FAILURE;
    }
    if (!in_pcd->HasNormals())
    {
        cerr << "no normals in " << pcd_path << endl;
        return EXIT_FAILURE;
    }
    // 2. translate the point cloud to the origin
    Eigen::Vector3d in_pcd_center = in_pcd->GetCenter();
    in_pcd->Translate(-in_pcd_center);

    // 3. seperate the non-horizontal and horizontal pcd
    double h_z_thresh = 0.95, 
           v_z_thresh = 0.5;
    int half_PI_bin_num = 100, 
        max_iter = 100;
    vector<EnumNormalGroup> normal_labels(input_pcd->points_.size(), EnumNormalGroup::other);
    PointCloud h_pcd, nh_pcd;
    pair<shared_ptr<PointCloud>, shared_ptr<PointCloud>> result = ClusterPointNormals(
        *input_pcd, h_pcd, nh_pcd, normal_labels, h_z_thresh, v_z_thresh);

    // 4. estimate the primary horizontal orientation
    double orientation = EstimateHorizontalOrientation(nh_pcd, half_PI_bin_num, max_iter);
    Eigen::AngleAxisd angle_axis(-orientation, Eigen::Vector3d(0,0,1));
    Eigen::Matrix3d rot_matrix = angle_axis.toRotationMatrix();
    Eigen::Vector3d rot_center = input_pcd->GetCenter();
    in_pcd->Rotate(rot_matrix, rot_center);
    nh_pcd.Rotate(rot_matrix, rot_center);

    size_t num_pts = nh_pcd.points_.size();
    // group normals
    vector<Eigen::Vector2d> normals_2d(num_pts, Eigen::Vector2d(0,0));
    for(size_t i = 0; i < num_pts; i++){
        normals_2d[i](0) = abs(nh_pcd.normals_[i][0]);
        normals_2d[i](1) = abs(nh_pcd.normals_[i][1]);
        normals_2d[i].normalized();
    }
    vector<Eigen::Vector2d> ng = {Eigen::Vector2d(0, 1), Eigen::Vector2d(1, 0)};
    size_t num_ng = ng.size();
    vector<size_t> ng_id(num_pts);
    vector<vector<size_t>> ng_pt_idx(num_ng);
    group_normals(normals_2d, ng, ng_id, ng_pt_idx);

    size_t selected_normal_id = (ng_pt_idx[0].size() > ng_pt_idx[1].size()) ? 0:1;
    cout << "selected primary normal group " << ng_pt_idx[selected_normal_id].size() << "/" << num_pts
            << "(" << ng[selected_normal_id][0] << ", " << ng[selected_normal_id][1] << ")" << endl;
    
    if (selected_normal_id>0){
        Eigen::AngleAxisd angle_axis(M_PI/2, Eigen::Vector3d(0,0,1));
        Eigen::Matrix3d rot_matrix = angle_axis.toRotationMatrix();
        Eigen::Vector3d rot_center = non_horizontal_pcd->GetCenter();
        in_pcd->Rotate(rot_matrix, rot_center);
        nh_pcd.Rotate(rot_matrix, rot_center);
        cout << "* rotated the align the primary normal with (1, 0, 0)" << endl;
    }
    cout << "rotated to align the axis. " << endl;
    
    // 6. save the whole aligned pcd and non-horitonatl aligned pcd
    string out_aa_pcd_path = working_folder + "/aligned.ply";
    string out_aa_nh_pcd_path = working_folder + "/non_horizontal.ply";
    open3d::io::WritePointCloud(out_aa_pcd_path, *in_pcd);
    open3d::io::WritePointCloud(out_aa_nh_pcd_path, nh_pcd);

    // 7. skeletonize the non-horizontal pcd
    string bss_atom_path = working_folder + "/bss_atom.txt";
    string bss_atom_pcd_path = working_folder + "/bss_atom_pcd.ply";
    
    fast_vote(
        nh_pcd, voxel_size,
        ng_pt_idx[selected_normal_id],
        bss_atom_path, bss_atom_pcd_path);
    
    return EXIT_SUCCESS;
}