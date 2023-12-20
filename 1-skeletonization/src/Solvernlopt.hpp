#include <nlopt.hpp>
#include <sys/time.h>
#include <iostream>
using namespace std;
class Solvernlopt{
private:
    nlopt::algorithm choose_alg(const int alg)
    {
        switch (alg)
        {
        case (0):
            // GOOD Cand
            return nlopt::algorithm::GN_DIRECT;
            break; 
        case (1):
            return nlopt::algorithm::LN_PRAXIS;
            break;
        case (2):
            // GOOD Cand
            return nlopt::algorithm::GN_MLSL_LDS;
            break;
        case (3):
            return nlopt::algorithm::LN_COBYLA;
            break;
        case (4):
            return nlopt::algorithm::LN_NEWUOA_BOUND;
            break;
        case (5):
            return nlopt::algorithm::LN_NELDERMEAD;
            break;
        case (6):
            return nlopt::algorithm::LN_SBPLX;
            break;
        case (7):
            return nlopt::algorithm::LN_AUGLAG;
            break;
        case (8):
            return nlopt::algorithm::LN_BOBYQA;
            break;
        case (9):
            return nlopt::algorithm::GN_DIRECT_L;
            break;
        case (10):
            return nlopt::algorithm::GN_DIRECT_L_RAND;
            break;
        case (11):
            return nlopt::algorithm::GN_DIRECT_NOSCAL;
            break;
        case (12):
            return nlopt::algorithm::GN_DIRECT_L_NOSCAL;
            break;
        case (13):
            return nlopt::algorithm::GN_DIRECT_L_RAND_NOSCAL;
            break;
        case (14):
            return nlopt::algorithm::GN_ORIG_DIRECT;
            break;
        case (15):
            return nlopt::algorithm::GN_ORIG_DIRECT_L;
            break;
        }
        return nlopt::algorithm::LN_NEWUOA_BOUND;
    }

    static double evaluate_model_nlopt (const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
    {
        auto tar_bin_float = x[0]/interval_;    // a virtual bin in float, range = [0, half_PI_bin_num * 2]
        double error = 0;

        for (size_t i = 0; i < half_PI_bin_num_; i++)
        {
            // abs diff in radian
            auto rot_diff = std::abs(tar_bin_float - i) * interval_;
            // if closer than +/- pi/4 ==> 
            error += std::cos(rot_diff) * std::sin(rot_diff) * frequency_[i]; 
        }
        return error;
    }; 

public: 
    static std::vector<double> frequency_;
    static double interval_;
    static int half_PI_bin_num_;
    Solvernlopt(std::vector<double> frequency, double interval, int half_PI_bin_num)
         {
             frequency_ = frequency;
             interval_ = interval;
             half_PI_bin_num_ = half_PI_bin_num;

         }

    
    double solve(const int alg = 4, const int max_eval = 100){
        std::vector<double> x(1);
        x[0] = M_PI/4; 
        double minf;
        nlopt::srand(20210509);
        // Other NLopt solvers: GN_DIRECT, LN_COBYLA, LN_NELDERMEAD, LN_NEWUOA_BOUND, LN_BOBYQA
        nlopt::opt opt(choose_alg(alg), 1);   // dimension = 1
        opt.set_lower_bounds(0.0);
        opt.set_upper_bounds(M_PI/2);
        opt.set_maxeval(max_eval);
        opt.set_min_objective(evaluate_model_nlopt, NULL);
        opt.set_xtol_rel(1e-6);
        struct timeval solver_start, solver_end;
        gettimeofday(&solver_start, NULL);
        // -------------- start solving -------
        try
        {
            nlopt::result result = opt.optimize(x, minf);
        }
        catch(std::exception &e) 
        {
            std::cout << "NLOpt failed: " << e.what() << std::endl;
        }
        // -------------- end of solving -------
        
        gettimeofday(&solver_end, NULL);
        long sec  = solver_end.tv_sec  - solver_start.tv_sec;
        long usec = solver_end.tv_usec - solver_start.tv_usec;
        long timecost = ((sec) * 1000 + usec/1000.0) + 0.5;
        
        cout << "* NLOPT: "
             << "\t best f = " << minf
             << "\t @ rot = " << x[0]
             << "\t opt time = " << timecost << " ms"
             << endl;
        return x[0];
    }
};