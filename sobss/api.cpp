#include "align.hpp"
#include "skeletonize.hpp"
#include "merge.hpp"

extern "C"{
    
void align(const char * pcd_path, const char * working_folder)
{   
    cout << "begin to align ... \n";
    align_pcd_to_y_axis(pcd_path, working_folder);
    cout << "aligned! \n";
}

void skeletonize(const char * working_folder)
{
    cout << "begin to collect_bss_atoms ... \n";
    collect_bss_atoms(working_folder);
    cout << "bss_atoms collected! \n";
}

void merge(const char * working_folder)
{
    cout << "begin to merge bss segms ... \n";
    merge_bss_segms(working_folder);
    cout << "bss segms merged! \n";
}

}