#include <fstream>	
#include <iostream> 
#include <string>
#include <vector>
#include <sstream> 
#include <strtk/strtk.hpp>
#include <cmath>
#include <typeinfo>
#include <memory>

std::fstream flogs("logs.txt", std::ios::app);

// Function that writes error messages to the file
void WriteError(const std::string& str) {
  static int count = 0;

  if (count == 0) {
	std::fstream ferrors("errors.txt", std::ios::out);
	ferrors << str << "\n";
	ferrors.close();
  } else {
	std::fstream ferrors("errors.txt", std::ios::app);
	ferrors << str << "\n";
	ferrors.close();
  }
  ++count;
}

/*  Struct to store one line (data for one sample) from train/test data files
 *	id: add id,
 *	features: hashed features,
 *	label: 1 if add was clicked, 0 otherwise.
 */
struct Row {
  std::string id;
  std::vector<size_t> features;
  double label;
};


/*	This function splits one line from train/test data file and returns vector of strings, 
 *	where each string corresponds to one feature
 */
std::vector<std::string> Split(const std::string& s) {

  static const char *delimeters = "\t\r\n\f;,";
  std::vector<std::string> tokens;

  if(!strtk::parse(s, delimeters, tokens)) {
	WriteError("Can't parse string: " + s);
  }
  return tokens;
}


/*	This function preprocess list of strings (list of features corresponding to one sample):
 *	add id is saved as string,
 *	sample features are converted from strings to integers using hash function,
 *	label is converted to float variable.
 *	Returns Row(id, hashed features, label)
 */
Row GetRow(const std::string& str, const std::vector<std::string>& headers, size_t size) {
	
  Row row;
  std::vector<std::string> tokens;
  static std::hash<std::string> hash_fun;

  tokens = Split(str);
  row.id = tokens[0];
  row.label = std::stod(tokens[1]);

  for(size_t i = 2; i != tokens.size(); ++i) {
	row.features.push_back(hash_fun(headers[i] + "_" + tokens[i])&(size-1));
  }
  return row;
}

// Parent class for FtrlProximal and FtrlProximal_approx classes
class Ftrl {
 public:
  double alpha;
  double beta;
  double l1;
  double l2;
  size_t size;

  std::vector<double> w_weights;
  std::vector<double> z_weights;
  std::vector<size_t> hash_idxs;

  virtual ~Ftrl(){};
  virtual void GetHashIndiciesToUpdate(const std::vector<size_t>& features) = 0;
  virtual void Update(const std::vector<size_t>& features, double prob, double label) = 0;
  virtual double Predict(const std::vector<size_t>& features) = 0;
};

/*	Class FtrlProximal implements Per-Coordinate FTRL-Proximal algorithm 
 *	with L1 and L2 regilarization for Logistic Regression
 */
class FtrlProximal: public Ftrl {
 public:
  std::vector<double> grad_sum;

  FtrlProximal(double alpha_v, double beta_v,	
  	           double l1_v, double l2_v, size_t size_v) {
    alpha = alpha_v;
    beta = beta_v;
	l1 = l1_v;
	l2 = l2_v;
	size = size_v;

	grad_sum.assign(size_v, 0.);
	w_weights.assign(size_v, 0.);
	z_weights.assign(size_v, 0.);
  }

  void GetHashIndiciesToUpdate(const std::vector<size_t>& features) {
  	// indicies interations can be added to hash_idxs if needed
    hash_idxs = features;
  }

  void Update(const std::vector<size_t>& features, double prob, double label) {
    double grad;
  	double sigma;

  	grad = prob - label;
  	GetHashIndiciesToUpdate(features);

  	for(auto it =  hash_idxs.begin(); it != hash_idxs.end(); ++it) {
  	  size_t i = *it;
  	  sigma = (sqrt(grad_sum[i] + grad * grad) - sqrt(grad_sum[i])) / alpha;
  	  z_weights[i] += grad - sigma * w_weights[i];
  	  grad_sum[i] += grad * grad;
  	}
  }

  double Predict(const std::vector<size_t>& features) {
    GetHashIndiciesToUpdate(features);

    double wf_product=0;
	for(auto it =  hash_idxs.begin(); it != hash_idxs.end(); ++it) {
  			
  	  size_t i = *it;
  	  int sign = (z_weights[i] < 0) ? -1 : 1;

  	  if (sign * z_weights[i] <= l1) {
  	    w_weights[i] = 0;
  	  } else {
  	    w_weights[i] = (sign * l1 - z_weights[i]) / ((beta + sqrt(grad_sum[i])) /alpha + l2);
  	  }

  	  wf_product += w_weights[i];
  	}
  	
  	return 1./(1. + exp(-std::max(std::min(wf_product, 35.), -35.)));
  	}

  ~FtrlProximal() {}
};


/*	Class FtrlProximal_simple implements Per-Coordinate FTRL-Proximal algorithm 
 *	with L1 and L2 regilarization for Logistic Regression.
 *
 *	We don't count and store grad_sum[i] as in FtrlProximal.
 *	Instead of that we approximate grad_sum[i] with function that depends on number of 
 *  positive and negative examples (pos_count and neg_count)
 */
class FtrlProximal_approx: public Ftrl {
 public:
  std::vector<size_t> pos_count;
  std::vector<size_t> neg_count;

  FtrlProximal_approx(double alpha_v, double beta_v, 
  	                  double l1_v, double l2_v, size_t size_v) {
    alpha = alpha_v;
	beta = beta_v;
	l1 = l1_v;
	l2 = l2_v;
	size = size_v;

	pos_count.assign(size_v, 0);
	neg_count.assign(size_v, 0);
	w_weights.assign(size_v, 0.);
	z_weights.assign(size_v, 0.);
  }

  void GetHashIndiciesToUpdate(const std::vector<size_t>& features) {
  	// indicies interations can be added to hash_idxs if needed
    hash_idxs = features; 
  }

  double ApproximateGradSum(size_t pos, size_t neg) {
    if (pos == 0 && neg == 0) { return 0;
	} else {
	  double pos_rate = pos / (pos + neg);
	  return pos * (1 - pos_rate) * (1 - pos_rate) + neg * pos_rate * pos_rate - pos_rate * neg;
	}
  }

  void Update(const std::vector<size_t>& features, double prob, double label) {
    double grad;
  	double sigma;

  	  grad = prob - label;
  	  GetHashIndiciesToUpdate(features);

  	  for(auto it =  hash_idxs.begin(); it != hash_idxs.end(); ++it) {
  	    size_t i = *it;
  		double approx(ApproximateGradSum(pos_count[i], neg_count[i]));

  		sigma = (sqrt(approx + grad*grad) - sqrt(approx)) / alpha;
  		z_weights[i] += grad - sigma * w_weights[i];
  		// grad_sum[i] += grad * grad;
  		pos_count[i] += label;
  		neg_count[i] += (1-label);
  	  }
  }

  double Predict(const std::vector<size_t>& features) {
    GetHashIndiciesToUpdate(features);

	double wf_product=0;
	for(auto it =  hash_idxs.begin(); it != hash_idxs.end(); ++it) {
  			
      size_t i = *it;
  	  int sign = (z_weights[i] < 0) ? -1 : 1;
  	  double approx(ApproximateGradSum(pos_count[i], neg_count[i]));

  	  if (sign * z_weights[i] <= l1) {w_weights[i] = 0;
  	  } else {
  		w_weights[i] = (sign * l1 - z_weights[i]) / ((beta + sqrt(approx)) /alpha + l2);
  	  }
  	  wf_product += w_weights[i];
  	}
  	return 1./(1. + exp(-std::max(std::min(wf_product, 35.), -35.)));
  		// return 1./(1. + exp(-wf_product));
  }

  ~FtrlProximal_approx() {}
};


double CountLogloss(double prob, double label) {
  double score;
  prob = std::max(std::min(prob, 1 - 1e-15), 1e-15);
  score = (label == 1.) ? -log(prob) : -log(1 - prob);
  return score;
}


int IsDegree2(long long x) {
  return x&&((x & (x - 1)) == 0);
}


void WriteValLoss(const std::string& filename, const std::vector<double>& val_loss) {
  // Save val scores
  std::fstream fvalloss(filename, std::ios::out);
  for(auto it = val_loss.begin(); it != val_loss.end(); ++it) {
    fvalloss << *it << ",";
  }
  fvalloss.close();
}


std::map<std::string, double> ReadParams(const std::string& filename) {
  std::fstream fparams(filename, std::ios::in);
  std::map<std::string, double> params;
  std::string param_name, param_value;

  if(!fparams.is_open()) {
    WriteError("Params file doesn't exist\n");
	exit(0);
  }

  while(std::getline(fparams, param_name, ' ')) {
    std::getline(fparams, param_value, '\n');
	params[param_name] = std::stod(param_value);
  }
  fparams.close();
  return params;
}


void Train(std::shared_ptr<Ftrl>& ftrl, int epoch_count, size_t size,
		   int holdout_period, std::vector<std::string>& headers,
	       const std::string& ftrain_name, const std::string& fvalloss_name) {
  flogs << "\nStart training...\n";
  size_t sample_count = 0;
  size_t val_sample_count = 0;
  double loss = 0.;

  Row row;
  std::string str;
  std::vector<double> val_loss;
  double prob;

  // Iterate by epochs, one epoch <=> one look through the whole train file
  for(int epoch = 0; epoch != epoch_count; ++epoch) {

    // Open train file
    std::fstream ftrain(ftrain_name, std::ios::in);    
    if(epoch ==0 && !ftrain.is_open()) {
	  WriteError("Train file doesn't exist");
	  return;
	}
  
    // Read headers of features
    std::getline(ftrain, str);
    headers = Split(str);
		
    //Process all lines below headers
    while (std::getline(ftrain, str)) {
      if(str == "") {
        continue;
      }
	
	  row = GetRow(str,  headers, size);
	  prob = ftrl->Predict(row.features);

	  if(sample_count > 0 && (sample_count % (holdout_period) == 0)) {
	    
	    ++val_sample_count;
	    loss += CountLogloss(prob, row.label);
	    val_loss.push_back(loss/val_sample_count);

	    if (IsDegree2(val_sample_count)) {
	      std::cout << "Epoch: "<< epoch << ", total samples: " << sample_count 
		            << ", val samples: " << val_sample_count 
		            << ", average val logloss: " << (loss/val_sample_count) 
		            << "\n";

		  flogs << "Epoch: "<< epoch << ", total samples: " << sample_count 
		        << ", val samples: " << val_sample_count 
			    << ", average val logloss: " << (loss/val_sample_count) 
			    << "\n";
				}	

	  } else { 
		ftrl->Update(row.features, prob, row.label);
	  }
    ++sample_count;    
    }  
  ftrain.close();  
  }
	
  WriteValLoss(fvalloss_name, val_loss);

  loss /= val_sample_count;
  flogs << "Train sample: " << sample_count << ", Validation sample: " << val_sample_count 
        << ", Validation loss: " << loss << "\n";
}


void Test(std::shared_ptr<Ftrl>& ftrl, size_t size,
		  std::vector<std::string>& headers,
	      const std::string& ftest_name, const std::string& fresult_name) {

  flogs << "\nStart Predicting...\n";
  Row row;
  std::string str;
  double prob;
	
  // Open test file
  std::fstream ftest(ftest_name, std::ios::in);
  if(!ftest.is_open()) {
    WriteError("Test file doesn't exist");
	return;
  }

  // Open file to write Predictions 
  std::fstream fresult(fresult_name, std::ios::out);
  fresult << "id,click\n";

  // Predict probabilities
  double test_loss = 0.;
  size_t counter = 0;
  while (std::getline(ftest, str)) {
    if(str == "") {
	  continue;
	}
	
	++counter;
	row = GetRow(str, headers, size);
	prob = ftrl->Predict(row.features);
	test_loss += CountLogloss(prob, row.label);

	fresult << row.id << "," << std::fixed << std::setprecision(13) << prob << "\n";
  }
	
  flogs << "Test samples: " << counter << ", Test loss: " << (test_loss/counter) << "\n";
  ftest.close();
  fresult.close();
}



int main(int argc, char* argv[]) {
	
  // Read arguments from command line
  if (argc < 5) {
    std::string argv0(argv[0]);
	std::string error = "Usage:" + argv0 + " fparams.txt ftrain.csv ftest.csv fresult.txt";
	WriteError(error);
	return 0;
  }

  // Create file instances
  std::string data_path("/media/julia/Data/BigData/");
  std::string fparams_name(argv[1]);
  std::string ftrain_name(data_path + argv[2]); 
  std::string ftest_name(data_path + argv[3]);
  std::string fresult_name(argv[4]);

  // Read parameters' values from the file 
  std::map<std::string, double> params;
  params = ReadParams(fparams_name);


  double l1(params["l1"]);
  double l2(params["l2"]);
  double alpha(params["alpha"]);
  double beta(params["beta"]);
  int epoch_count(params["epoch_count"]);
  size_t size(params["size"]); // size = 2^power
  int holdout_period(params["holdout_period"]);
  int is_approx(params["is_approx"]);


  flogs << "\nParameters:\n";
  for (std::map<std::string, double>::iterator it = next(params.begin()); it != params.end(); ++it) {
    flogs << "\t" << it->first << ": " << std::fixed  << it->second << '\n';
  }

  // Create FtrlProximal instance
  std::shared_ptr<Ftrl> ftrl;
  if (is_approx) { 
    ftrl.reset(new FtrlProximal(alpha, beta, l1, l2, size));
  } else {
  	ftrl.reset(new FtrlProximal_approx(alpha, beta, l1, l2, size));
  }	
  flogs << "\nFtrlProximal created\n";
  
  // Train and test
  std::vector<std::string> headers;
  Train(ftrl, epoch_count, size, holdout_period, headers, ftrain_name, "val_loss.txt");
  Test(ftrl, size, headers, ftest_name, fresult_name);

  flogs.close();
  return 0;
}

