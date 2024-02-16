#include "skeletonize.hpp"
#include "merge.hpp"

extern "C"{

void skeletonize(const char * pcd_path, const char * working_folder)
{
    cout << "begin to collect_bss_atoms ... \n";
    string pcd_path_str(pcd_path);
    string working_folder_str(working_folder);
    collect_bss_atoms(pcd_path_str, working_folder_str);
    cout << "bss_atoms collected! \n";
}

void merge(const char * working_folder)
{
    cout << "begin to merge bss segms ... \n";
    merge_bss_segms(working_folder);
    cout << "bss segms merged! \n";
}

}