# pragma once

#include "util.hpp"

int merge_bss_segms(
    const string working_folder) 
{
    filesystem::path bss_segm_path = filesystem::path(working_folder) / filesystem::path("bss_coarse_segm.txt");
    if (!filesystem::exists(bss_segm_path.string()))
    {
        cerr << "no bss_segm.txt in " << working_folder << endl;
        return EXIT_FAILURE;
    }
    
    filesystem::path conf_path = fileseystem::path(working_folder) / fileseystem::path("config.json");
    if (!filesystem::exists(conf_path.string()))
    {
        cerr << "no config.json in " << working_folder << endl;
        return EXIT_FAILURE;
    }

    rj::Document conf_doc;
    if (!read_config(conf_path.string().c_str(), conf_doc))
        return EXIT_FAILURE;
    
    double lambda;
    if (conf_doc.HasMember("lambda") && conf_doc["lambda"].IsDouble())
        lambda = conf_doc["lambda"].GetDouble();
    else
    {
        cerr << "lambda is not set in " << conf_path.string() << endl;
        cout << "set lambda as 0.5 by default. \n";
        lambda = 0.5;
    }

    double sigma;
    if (conf_doc.HasMember("sigma") && conf_doc["sigma"].IsDouble())
        sigma = conf_doc["sigma"].GetDouble();
    else
    {
        cerr << "sigma is not set in " << conf_path.string() << endl;
        cout << "set sigma as 0.5 by default. \n";
        sigma = 0.5;
    }

    // 1. load in the result of coarse segmentation
    vector<vector<double>> input_vol_params;
    ifstream file(bss_segm_path.string());
    string str;
    size_t num_vol, num_box;
    num_box = 0;
    while(getline(file, str))
    {
        vector<double> v;
        istringstream iss(str); 
        copy(istream_iterator<double>(iss),
            istream_iterator<double>(),
            back_inserter(v));
        assert(v.size() == 8);
        input_vol_params.push_back(v);
        if (v[6] == 1)
            num_box ++;
    }
    num_vol = input_vol_params.size();
    cout << num_vol << " volumes to merge! \n";
    // 2. precompute
    Eigen::MatrixXd merging_cost(num_vol, num_vol+1);
    Eigen::MatrixXi merge_in_out(num_vol, num_vol);
    vector<Eigen::MatrixXd> all_8pts_vol(num_vol);
    Eigen::VectorXd min_allx(num_vol), max_allx(num_vol), 
                    min_ally(num_vol), max_ally(num_vol), 
                    min_allz(num_vol), max_allz(num_vol);
    vector<size_t> num_pairs;
    double total_volume = precomputing_merging_cost(
        input_vol_params, num_box, merging_cost, merge_in_out, all_8pts_vol,
        min_allx, max_allx, min_ally, max_ally, min_allz, max_allz,
        num_pairs, sigma);
    
    // 3. merge
    Eigen::MatrixXd is_mergedto(num_vol, num_vol+1);
    binary_optimization(num_vol, merging_cost, 
            lambda, total_volume, is_mergedto);
    cout << "*** Optimization done!" << endl;

    // 4. parse the optimization result
    vector<vector<double>> merged_vol_param;
    vector<Eigen::MatrixXd> merged_vol_8pts;
    parsing_merging_result(num_vol, input_vol_params, 
        is_mergedto, merge_in_out, all_8pts_vol,
        min_allx, max_allx, min_ally, max_ally, min_allz, max_allz,
        merged_vol_param, merged_vol_8pts);
    
    // 5. union the merged result
    vector<Nef_Polyhedron> merged_vol_n_polyhedra;
    Nef_Polyhedron union_merged;
    gen_all_meshes(merged_vol_param, merged_vol_n_polyhedra);
    if (merged_vol_n_polyhedra.size() > 0)
        size_t union_steps = tree_union(merged_vol_n_polyhedra, union_merged);
    
    // 6. save the result
    filesystem::param_path = filesystem::path(working_folder) / filesystem::path("bss_merged_segm.txt");
    save_vol_param(merged_vol_param, param_path.string());

    vector<Point_3> union_points_tri;
    vector<vector<size_t>> union_polygons_tri; 
    if (merged_vol_n_polyhedra.size() > 0)
    {
        CGAL::convert_nef_polyhedron_to_polygon_soup(
            union_merged, union_points_tri, union_polygons_tri, true);
    }

    filesystem::path obj_path = filesystem::path(working_folder) / filesystem::path("bss_merged_vol.obj");  
    save_vol_8pts_as_obj(merged_vol_8pts, obj_path.string());
    filesystem::path union_tri_path = filesystem::path(working_folder) / filesystem::path("bss_merged_tri.obj");
    CGAL::IO::write_OBJ(union_tri_path.string(), union_points_tri, union_polygons_tri);
    
    return EXIT_SUCCESS;
}