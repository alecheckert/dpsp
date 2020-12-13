/*
gsdpdiffdefoc.cpp -- Gibbs sampler for a Dirichlet process over
log diffusivities, given a set of trajectories.

This function takes a specific CSV input that must be formatted
by upstream programs. The input (*in_csv*) is a CSV with the 
following specification:

    - Every line in *in_csv* corresponds to one observed trajectory.
    - The first element in each line is the sum of 2D radial
        squared displacements for that trajectory (a float).
    - The second element in each line is the number of displacements
        in that trajectory (an int).
    - There is no header line.

After running Gibbs sampling to estimate the posterior diffusivity
distribution given a Dirichlet process mixture model, this program
writes the output distribution to a CSV (*output_csv*) with the 
following specification:

    - Each line in *output_csv* corresponds to a single count of 
        a diffusivity component - that is, one Markov chain's 
        position at one iteration.
    - The first element in each line is log(4 * t * diffusivity),
        where *t* is the frame interval of the microscope, and 
        *diffusivity* is the diffusivity of the component in 
        um^2 s^-1.
    - The second element in each line is the number of displacements
        currently assigned to that observation.
    - There is no header line.

*/
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>

#include <cmath>
#include <ctime>

#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <libgen.h>

#define OPTSTR "a:n:b:m:t:s:c:d:z:e:x:l:i:vh"
#define USAGE_FMT " [-a alpha] [-n n_iter] [-b burnin] [-m n_aux] [-t frame_interval] [-s metropolis_sigma] [-c min_log_D] [-d max_log_D] [-z buffer_size] [-e seed] [-x max_occ_weight] [-l loc_error] [-i bias_csv] [-v verbose] [-h help] IN_CSV OUT_CSV"

void usage(char *progname, int opt);

double get_max(int size, double *arr);

void vecsub(int size, double *arr, double arg);

int random_category(int size, double *weights, double p);

int argmin(int size, bool *active);

double argmatch(int size, double value, double *refs, double *rets);

int main (int argc, char *argv[]) {

    // Opt parsing
    double alpha = 2.0;
    int n_iter = 10;
    int burnin = 20;
    int m = 10;
    int m0 = 30;
    int seed = 113;
    double frame_interval = 0.01;
    double metropolis_sigma = 0.1;
    int B = 10000;
    double min_log_D = -2.0;
    double max_log_D = 2.0;
    double max_log_L, p;
    bool verbose = false;
    int max_occ_weight = 5;
    double loc_error = 0.0;
    char bias_csv [300];
    bool correct_bias = false;
    int metropolis_steps_per_iter = 5;

    int opt;
    while (( opt = getopt(argc, argv, OPTSTR)) != -1) {
        switch (opt) {
            case 'a':
                alpha = std::stod(optarg);
                break;
            case 'n':
                n_iter = std::stoi(optarg);
                break;
            case 'b':
                burnin = std::stoi(optarg);
                break;
            case 'm':
                m = std::stoi(optarg);
                break;
            case 't':
                frame_interval = std::stod(optarg);
                break;
            case 's':
                metropolis_sigma = std::stod(optarg);
                break;
            case 'c':
                min_log_D = std::stod(optarg);
                break;
            case 'd':
                max_log_D = std::stod(optarg);
                break;
            case 'z':
                B = std::stoi(optarg);
                break;
            case 'e':
                seed = std::stoi(optarg);
                break;
            case 'x':
                max_occ_weight = std::stoi(optarg);
                break;
            case 'l':
                loc_error = std::stod(optarg);
                break;
            case 'i':
                std::strcpy(bias_csv, optarg);
                correct_bias = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
            default:
                usage(basename(argv[0]), opt);
                break;
        }
    }

    // Check that the user has specified two positional arguments
    if ((argc - optind) < 2) {
        usage(basename(argv[0]), opt);
    }
    
    // Get the input CSV path, the first positional argument
    char *in_csv = argv[optind];

    // Get the output CSV path, the second positional argument
    char *out_csv = argv[optind+1];

    // Initialize some variables
    std::ifstream f_in;
    std::ofstream f_out;
    int n_tracks = 0;
    int i, j, ci, n_active, track_n_disps;
    int str_buffer_size = 2000;
    char *str_buffer = new char [str_buffer_size];
    char *pch;
    int *components, *n_disps;
    double *ss, *log_d_corr_val, *bias_terms;
    double log_L_ratio, exp_proposed, track_ss, proposed, proposed_with_err, curr, log_p;

    // 2 * (1D spatial variance) due to localization error
    double le2 = 4 * loc_error * loc_error;

    // Factor necessary for the Metropolis acceptance distribution
    double sqrt_metropolis_var = std::sqrt(2.0) * metropolis_sigma;

    // Prior distribution
    double min_log_4Dt = min_log_D + std::log(4.0 * frame_interval);
    double max_log_4Dt = max_log_D + std::log(4.0 * frame_interval);
    std::default_random_engine generator {(unsigned int) seed};
    std::uniform_real_distribution<> prior(min_log_4Dt, max_log_4Dt);

    // Generate random real numbers between 0 and 1
    std::uniform_real_distribution<> random_prob(0.0, 1.0);

    // Normally distributed random variable centered at 0 with standard
    // deviation equal to *metropolis_sigma*
    std::normal_distribution<double> metropolis_nudge(0.0, metropolis_sigma);

    // Parse the input CSV with trajectory info
    f_in.open(in_csv);
    if (f_in.is_open()) {
        
        // Get the number of lines in the CSV
        f_in.getline(str_buffer, str_buffer_size);
        while (f_in) {
            n_tracks++;
            f_in.getline(str_buffer, str_buffer_size);
        }
        if (verbose) {
            std::cout << "n_tracks = " << n_tracks << std::endl;
        }
        f_in.close(); f_in.open(in_csv);

        // The mixture components to which each trajectory is 
        // currently assigned. For example, if components[i] = ci,
        // then trajectory i is currently assigned to component ci.
        components = new int [n_tracks];

        // The sum of squared 2D radial displacements for each
        // trajectory
        ss = new double [n_tracks];

        // Number of displacements per trajectory
        n_disps = new int [n_tracks];

        // Read every trajectory in the file into the arrays
        i = 0;
        f_in.getline(str_buffer, str_buffer_size);
        while (f_in) {
            pch = strtok(str_buffer, ",");
            ss[i] = std::stod(pch);
            pch = strtok(NULL, ",");
            n_disps[i] = std::stoi(pch);
            f_in.getline(str_buffer, str_buffer_size);
            i++;
        }
        f_in.close();

    } else {
        std::cerr << "Could not read path " << in_csv << std::endl;
        delete [] str_buffer;
        return 1;
    }

    // Read the bias CSV if applicable. These are essentially modifiers
    // to the probability of each component that depend on its diffusivity.

    // Probability for each trajectory to start a new component at each 
    // iteration
    double branch_prob = alpha / (alpha + (double) n_tracks);

    // Occupations of each component, defined as the number of displacements
    // that are currently assigned to each component. For example, if 
    // c_occs[ci] = 0, then there are no trajectories currently assigned
    // to component ci.
    int *c_occs = new int [B];

    // Corresponding log occupations, so we don't have to call logarithms
    // at each step
    double *c_log_occs = new double [B];

    // Set all occupations to zero initially
    for (ci=0; ci<B; ci++) {
        c_occs[ci] = 0;
    }

    // Current log (4 * frame_interval * diffusivities) for each component.
    // For example, if c_params[ci] = -2.0, then
    // log (4 * frame_interval * diffusivity[ci]) = -2.0.
    double *c_params = new double [B];

    // exp(-log (4 * frame_interval * diffusivities)), to save time spent
    // recomputing this for every trajectory
    double *c_exp_params = new double [B];

    // Whether each component is currently active - that is, whether there
    // are any trajectories/displacements currently assigned to it.
    // For example, if c_active[ci] = False, then component ci is not 
    // currently active.
    bool *c_active = new bool [B];

    // A flexible array that holds indices for components
    int *c_indices = new int [B];

    // Read in the state bias CSV, if applicable
    int n_D_corr_val = 0;
    if (correct_bias) {
        f_in.open(bias_csv);
        if (f_in) {

            // Get the number of lines in the file
            f_in.getline(str_buffer, str_buffer_size);
            while (f_in) {
                n_D_corr_val++;
                f_in.getline(str_buffer, str_buffer_size);
            }
            f_in.close(); f_in.open(bias_csv);

            // The log (4 * (t * diffusivities + loc_error**2)) to which
            // each bias term corresponds
            log_d_corr_val = new double [n_D_corr_val];

            // The bias terms
            bias_terms = new double [n_D_corr_val];

            // Read the log diffusivities and bias terms
            i = 0;
            f_in.getline(str_buffer, str_buffer_size);
            while (f_in) {
                pch = strtok(str_buffer, ",");
                log_d_corr_val[i] = std::log(4 * frame_interval * std::stod(pch) + le2);
                pch = strtok(NULL, ",");
                bias_terms[i] = std::log(std::stod(pch));
                f_in.getline(str_buffer, str_buffer_size);
                i++;
            }
            f_in.close();
        } else {
            correct_bias = false;
        }
    }


    // INITIAL COMPONENT ASSIGNMENTS

    // Activate *m0* initial components for this mixture
    for (i=0; i<m0; i++) {
        c_params[i] = std::log(std::exp(prior(generator)) + le2);
        c_exp_params[i] = std::exp(-c_params[i]);
    }

    // Assign each trajectory randomly to one of the initial components.
    // The probability to assign trajectory i to component ci is 
    // proportional to the likelihood of c_params[ci] given trajectory i.
    double *log_L = new double [m0];
    double *L = new double [m0];
    for (i=0; i<n_tracks; i++) {

        // Calculate the likelihood of each of the log diffusivities 
        // given this trajectory
        if (correct_bias) {
            for (ci=0; ci<m0; ci++) {
                log_L[ci] = -ss[i] * c_exp_params[ci] - 
                    n_disps[i] * c_params[ci] - 
                    argmatch(n_D_corr_val, c_params[ci], log_d_corr_val, bias_terms);
            }
        } else {
            for (ci=0; ci<m0; ci++) {
                log_L[ci] = -ss[i] * c_exp_params[ci] - 
                    n_disps[i] * c_params[ci];
            }
        }

        // Regularize the log likelihood
        max_log_L = get_max(m0, log_L);
        vecsub(m0, log_L, max_log_L);

        // Convert from log likelihood to likelihood
        for (ci=0; ci<m0; ci++) 
            L[ci] = std::exp(log_L[ci]);

        // Sample from the resulting distribution
        p = random_prob(generator);
        components[i] = random_category(m0, L, p);

        // Record the number of displacements from this trajectory
        c_occs[components[i]] += n_disps[i];

    }
    delete [] log_L;
    delete [] L;

    // Determine which components start out active
    for (ci=0; ci<B; ci++) {
        if (c_occs[ci] > 0) {
            c_active[ci] = true;
            if (correct_bias) {
                c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci])) -
                    argmatch(n_D_corr_val, c_params[ci], log_d_corr_val, bias_terms);
            } else {
                c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci]));
            }
        } else {
            c_active[ci] = false;
            c_log_occs[ci] = 0;
        }
    }

    // General-purpose arrays for sampling
    double *candidate_params = new double [B];
    double *candidate_log_L = new double [B];
    double *candidate_L = new double [B];

    // Open the output file
    f_out.open(out_csv);


    // CORE GIBBS SAMPLING ROUTINE

    for (int iter_idx = 0; iter_idx < n_iter; iter_idx++) {

        // For each trajectory, either start a new component or assign to 
        // one of the existing components
        for (i=0; i<n_tracks; i++) {

            // Component to which this trajectory is currently assigned
            ci = components[i];
        
            // Sum squared displacement and number of displacements for this
            // trajectory
            track_ss = ss[i];
            track_n_disps = n_disps[i];

            // Remove the present trajectory from the component occupations
            c_occs[ci] -= track_n_disps;

            // If there are no more observations left for that component,
            // remove it from consideration
            if (c_occs[ci] == 0) {
                c_active[ci] = false;
            } else {
                if (correct_bias) {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci])) - 
                        argmatch(n_D_corr_val, c_active[ci], log_d_corr_val, bias_terms);
                } else {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci]));
                }
            }

            // Choose whether to have this trajectory start a new Markov chain
            p = random_prob(generator);
            if (p < branch_prob) {

                // Give this new component the lowest unassigned component index
                ci = argmin(B, c_active);

                // If this exceeds the buffer size, bail
                if ((ci == -1) or (ci >= B-1)) {
                    delete [] str_buffer;
                    delete [] c_occs;
                    delete [] c_log_occs;
                    delete [] c_params;
                    delete [] c_exp_params;
                    delete [] c_active;
                    delete [] c_indices;
                    delete [] components;
                    delete [] ss;
                    delete [] n_disps;
                    delete [] candidate_params;
                    delete [] candidate_log_L;
                    delete [] candidate_L;
                    if (correct_bias) {
                        delete [] log_d_corr_val;
                        delete [] bias_terms;
                    }
                    f_out.close();
                    std::cout << "\nWARNING: Buffer size (" << B << ") exceeded; terminating iteration\n";
                    return 1;
                }

                // Record the new component's index and the initial number of jumps
                // assigned to it
                c_active[ci] = true;
                components[i] = ci;
                c_occs[ci] = track_n_disps;

                // Choose the new component's diffusivity. Consider *m* candidates
                // drawn from the prior, and weight each candidate by its likelihood
                // given the present trajectory.
                for (j=0; j<m; j++) {
                    candidate_params[j] = std::log(std::exp(prior(generator)) + le2);
                    candidate_log_L[j] = -track_ss * std::exp(-candidate_params[j]) - 
                        track_n_disps * candidate_params[j];
                }

                // Regularize the log likelihood
                max_log_L = get_max(m, candidate_log_L);
                vecsub(m, candidate_log_L, max_log_L);

                // Convert from log likelihood to likelihood
                for (j=0; j<m; j++)
                    candidate_L[j] = std::exp(candidate_log_L[j]);

                // Sample from the resulting distribution and record the 
                // corresponding log diffusivity
                p = random_prob(generator);
                j = random_category(m, candidate_L, p);
                c_params[ci] = candidate_params[j];
                c_exp_params[ci] = std::exp(-candidate_params[j]);

                // Update the log occupation with a bias term, if applicable
                if (correct_bias) {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, track_n_disps)) -
                        argmatch(n_D_corr_val, c_params[ci], log_d_corr_val, bias_terms);
                } else {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, track_n_disps));
                }

            } else {

                // Assign to an existing component. The probability to assign
                // trajectory i to component ci is proportional to the
                // likelihood of the log diffusivity c_params[ci] given trajectory
                // i and given the current occupation of component ci.
                n_active = 0;
                for (ci=0; ci<B; ci++) {
                    if (c_active[ci]) {
                        c_indices[n_active] = ci;
                        candidate_log_L[n_active] = -track_ss * c_exp_params[ci] - 
                            track_n_disps * c_params[ci] + c_log_occs[ci];
                        n_active++;
                    }
                }

                // Regularize
                max_log_L = get_max(n_active, candidate_log_L);
                vecsub(n_active, candidate_log_L, max_log_L);

                // Convert from log likelihood to likelihood
                for (j=0; j<n_active; j++) {
                    candidate_L[j] = std::exp(candidate_log_L[j]);
                }

                // Assign this trajectory to a random component
                p = random_prob(generator);
                ci = c_indices[random_category(n_active, candidate_L, p)];
                components[i] = ci;

                // Update the occupation count for this component
                c_occs[ci] += track_n_disps;
                if (correct_bias) {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci])) - 
                        argmatch(n_D_corr_val, c_params[ci], log_d_corr_val, bias_terms);
                } else {
                    c_log_occs[ci] = std::log(std::min(max_occ_weight, c_occs[ci]));
                }
            }
        }

        // For each of the active components, resample the corresponding 
        // log diffusivity using a Metropolis-Hastings step.
        for (ci=0; ci<B; ci++) {
            if (c_active[ci]) {

                for (int step_idx=0; step_idx<metropolis_steps_per_iter; step_idx++) {

                    // Current log diffusivity of this parameter
                    curr = c_params[ci];

                    // Add a random nudge, making sure not to go outside
                    // the minimum or maximum permissible diffusivities
                    // proposed = curr + metropolis_nudge(generator);
                    proposed = std::log(std::exp(curr) - le2) + metropolis_nudge(generator);
                    while ((proposed < min_log_4Dt) or (proposed > max_log_4Dt)) {
                        // proposed = curr + metropolis_nudge(generator);
                        proposed = std::log(std::exp(curr) - le2) + metropolis_nudge(generator);
                    }

                    // Account for localization error
                    proposed_with_err = std::log(std::exp(proposed) + le2);
                    exp_proposed = std::exp(-proposed_with_err);

                    // Compute the log likelihood ratio between the proposed
                    // and current diffusivity
                    log_L_ratio = 0.0;
                    // if (correct_bias) {
                    //     log_L_ratio = + argmatch(n_D_corr_val, proposed, log_d_corr_val, bias_terms) -  
                    //         argmatch(n_D_corr_val, c_params[ci], log_d_corr_val, bias_terms);
                    // } else {
                    //     log_L_ratio = 0.0;
                    // }

                    for (i=0; i<n_tracks; i++) {
                        if (components[i] == ci) {
                            log_L_ratio += -ss[i] * exp_proposed - n_disps[i] * proposed_with_err;
                            log_L_ratio -= (-ss[i] * c_exp_params[ci] - n_disps[i] * c_params[ci]);
                        }
                    }

                    // Modify the log likelihood ratio to make the chain ergodic on 
                    // the limited support
                    log_L_ratio += std::log((std::erf((max_log_4Dt - curr) / sqrt_metropolis_var) - 
                        std::erf((min_log_4Dt - curr) / sqrt_metropolis_var)));
                    log_L_ratio -= std::log((std::erf((max_log_4Dt - proposed) / sqrt_metropolis_var) - 
                        std::erf((min_log_4Dt - proposed) / sqrt_metropolis_var)));                   

                    // Determine whether to accept the update
                    p = std::log(random_prob(generator));
                    if (log_p < log_L_ratio) {
                        c_params[ci] = proposed_with_err;
                        c_exp_params[ci] = exp_proposed;
                    }
                }
            }
        }

        // Record the current log diffusivity for all active components and
        // the corresponding weight
        if (iter_idx > burnin) {
             for (ci=0; ci<B; ci++) {
                if (c_active[ci]) {
                    f_out << c_params[ci] << "," << c_occs[ci] << "\n";
                }
             } 
        }

        if (verbose) {
            std::cout << "Finished with " << iter_idx << "/" << 
                n_iter << " iterations, with " << n_active << 
                " MCs active\r";
            std::cout.flush();
        }
    }

    f_out.close();

    // Deallocate
    delete [] str_buffer;
    delete [] c_occs;
    delete [] c_log_occs;
    delete [] c_params;
    delete [] c_exp_params;
    delete [] c_active;
    delete [] c_indices;
    delete [] components;
    delete [] ss;
    delete [] n_disps;
    delete [] candidate_params;
    delete [] candidate_log_L;
    delete [] candidate_L;
    if (correct_bias) {
        delete [] log_d_corr_val;
        delete [] bias_terms;
    }
    return 0;
}

void usage(char *progname, int opt) {
    std::cerr << progname << USAGE_FMT << std::endl;
    std::exit(1);
}

/*
 *  function: get_max
 *  -----------------
 *  Return the maximum in an array of doubles.
 *
*/
double get_max(int size, double *arr) {
    double curr = -10000.0;
    for (int i=0; i<size; i++) {
        if (arr[i] > curr)
            curr = arr[i];
    }
    return curr;
}

/*
 *  function: vecsub
 *  ----------------
 *  Subtract a double argument from each element of an array.
 *
*/
void vecsub(int size, double *arr, double arg) {
    for (int i=0; i<size; i++) {
        arr[i] -= arg;
    }
}

/*
 *  function: random_category
 *  -------------------------
 *  Randomly select an integer I from 0 to size-1, with
 *  the probability for each I proportional to weights[I].
 *
 *  args
 *  ----
 *    size      :   the number of categories
 *    weights   :   the array of weights
 *    p         :   a random real number between 0 and 1
 * 
 *  returns
 *  -------
 *    I, the category
 *
*/
int random_category(int size, double *weights, double p) {
    int j;

    // Calculate the unnormalized CDF
    for (j=1; j<size; j++) 
        weights[j] += weights[j-1];

    // Normalize
    p *= weights[size-1];

    // Inverse CDF sampling
    for (j=0; j<size; j++) {
        if (p < weights[j])
            return j;
    }
    return size-1;
}

/*
 *  function: argmin
 *  ----------------
 *  Return the lowest index ci such that active[ci] is true.
 *
*/
int argmin(int size, bool *active) {
    for (int i=0; i<size; i++) {
        if (not active[i]) 
            return i;
    }
    return -1;
}

/*
 *  function: argmatch
 *  ------------------
 *  Given a strictly increasing set of reference numbers (*refs*), a
 *  set of return numbers (*refs*), and a particular query (*value*),
 *  find the highest index *i* such that refs[i] < value. Then return
 *  *rets[i]*.
 *
 *  args
 *  ----
 *    size      :   the number of elements in *refs* and *rets*
 *    value     :   query value
 *    refs      :   reference values to compare against query
 *    rets      :   possible return values
 *
 *  returns
 *  -------
 *    the element in *rets* matching the condition in the docstring
 *      above
 *
*/
double argmatch(int size, double value, double *refs, double *rets) {
    for (int i=1; i<size; i++) {
        if (value < refs[i]) {
            return rets[i-1];
        }
    }
    return rets[size-1];
}
