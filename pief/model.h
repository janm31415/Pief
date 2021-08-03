#pragma once

#include <vector>
#include <stdint.h>
#include <string>

namespace jtk
  {
  class buffer_object;
  class vertex_array_object;
  }

enum lifting_step_type
  {
  lst_predict,
  lst_update,
  lst_scale_even,
  lst_scale_odd
  };

struct lifting_step
  {
  lifting_step_type type;
  std::vector<double> mask;
  };

std::vector<lifting_step> parse(const std::string& wavelet_rules);

enum scheme
  {
  jamlet_linear,
  jamlet_quadratic,
  jamlet_cubic,
  jamlet_4_point,
  cdf_5_3,
  cdf_9_7,
  chaikin,
  cubic_bsplines,
  cubic_bspline_wavelets,
  daubechies_d4,
  four_point,
  haar,
  custom
  };

struct model
  {
  model();
  ~model();

  void delete_render_objects();

  int levels;
  std::vector<double> values;

  jtk::vertex_array_object* _vao;
  jtk::buffer_object* _vbo_array;
  };

void biorthogonal_inverse(double* sample, uint64_t n, uint64_t level, scheme s, const std::vector<lifting_step>& custom_steps);

void make_scaling_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps);
void make_wavelet_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps);
void make_biorthogonal_scaling_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps);
void make_biorthogonal_wavelet_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps);
void make_test_function(model& m, int f);

void fill_render_data(model& m, const std::vector<double>& values);

double compute_volume(const std::vector<double>& values);

void get_spline_component(std::vector<double>& values, const model& m, int _level, scheme s, const std::vector<lifting_step>& custom_steps);
void get_wavelet_component(std::vector<double>& values, const model& m, int _level, scheme s, const std::vector<lifting_step>& custom_steps);

double compress(model& m, double threshold, scheme s, const std::vector<lifting_step>& custom_steps);
void smooth(model& m, double threshold, int smooth_level, scheme s, const std::vector<lifting_step>& custom_steps);

void analyze(scheme s, const std::vector<lifting_step>& custom_steps);

double compute_smoothness(const std::vector<double>& samples, double scale = 1.0);

void construct_stable_wavelet(std::vector<lifting_step>& custom_steps, double& sob, double& sob_dual);