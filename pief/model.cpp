#include "model.h"
#include "logging.h"

#include "buffer_object.h"
#include "vertex_array_object.h"

#include "parse.h"

#include "../lifting/lifting.h"
#include "../lifting/sobolev.h"

#include <algorithm>
#include <sstream>
#include <numeric>

namespace
  {
  void gl_check_error(const char *txt)
    {
    unsigned int err = glGetError();
    if (err)
      {
      std::stringstream str;
      str << "GL error " << err << ": " << txt;
      throw std::runtime_error(str.str());
      }
    }

  void forward_custom(double* sample, uint64_t n, uint64_t level, const std::vector<lifting_step>& custom_steps)
    {
    using namespace lifting;
    for (const auto& s : custom_steps)
      {
      switch (s.type)
        {
        case lst_predict: predict(sample, n, s.mask, level, 1, false); break;
        case lst_update: update(sample, n, s.mask, level, 1, false); break;
        case lst_scale_even: if (!s.mask.empty()) scale_even(sample, n, s.mask.front(), level, 1, 1, false); break;
        case lst_scale_odd:  if (!s.mask.empty()) scale_odd(sample, n, s.mask.front(), level, 1, 1, false); break;
        }
      }
    }

  void inverse_custom(double* sample, uint64_t n, uint64_t level, const std::vector<lifting_step>& custom_steps)
    {
    using namespace lifting;
    for (auto rit = custom_steps.rbegin(); rit != custom_steps.rend(); ++rit)
      {
      switch (rit->type)
        {
        case lst_predict: ipredict(sample, n, rit->mask, level, 1, false); break;
        case lst_update: iupdate(sample, n, rit->mask, level, 1, false); break;
        case lst_scale_even: if (!rit->mask.empty()) iscale_even(sample, n, rit->mask.front(), level, 1, 1, false); break;
        case lst_scale_odd: if (!rit->mask.empty()) iscale_odd(sample, n, rit->mask.front(), level, 1, 1, false); break;
        }
      }
    }

  void inverse_custom_biorthogonal(double* sample, uint64_t n, uint64_t level, const std::vector<lifting_step>& custom_steps)
    {
    using namespace lifting;
    for (auto rit = custom_steps.rbegin(); rit != custom_steps.rend(); ++rit)
      {
      switch (rit->type)
        {
        case lst_predict: iupdate(sample, n, rit->mask, level, 1, false); break;
        case lst_update: ipredict(sample, n, rit->mask, level, 1, false); break;
        case lst_scale_even: if (!rit->mask.empty()) iscale_even(sample, n, 1.0 / rit->mask.front(), level, 1, 1, false); break;
        case lst_scale_odd: if (!rit->mask.empty()) iscale_odd(sample, n, 1.0 / rit->mask.front(), level, 1, 1, false); break;
        }
      }
    }

  int get_width(scheme s)
    {
    switch (s)
      {
      case jamlet_linear:
        return 4;
      case jamlet_quadratic:
        return 5;
      case jamlet_cubic:
        return 4;
      case jamlet_4_point:
        return 5;
      case cdf_5_3:
        return 4;
      case cdf_9_7:
        return 4;
      case cubic_bsplines:
        return 4;
      case cubic_bspline_wavelets:
        return 4;
      case daubechies_d4:
        return 4;
      case haar:
        return 4;
      case four_point:
        return 4;
      case chaikin:
        return 4;
      case custom:
        return 5;
      default:
        return 4;
      }
    }

  }

model::model() : levels(12), _vao(nullptr), _vbo_array(nullptr)
  {

  }

model::~model()
  {
  delete_render_objects();
  }

void model::delete_render_objects()
  {
  if (_vao)
    {
    _vao->release();
    delete _vao;
    _vao = nullptr;
    }
  if (_vbo_array)
    {
    _vbo_array->release();
    delete _vbo_array;
    _vbo_array = nullptr;
    }
  }

void biorthogonal_inverse(double* sample, uint64_t n, uint64_t level, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  switch (s)
    {
    case jamlet_linear:
      ipredict(sample, n, get_update_mask_jamlet_linear(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_jamlet_linear(), level, 1, false);
      break;
    case jamlet_quadratic:
      ipredict(sample, n, get_second_update_mask_jamlet_quadratic(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_jamlet_quadratic(), level, 1, false);
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_jamlet_quadratic(), level, 1, 1, false);
      ipredict(sample, n, get_first_update_mask_jamlet_quadratic(), level, 1, false);
      break;
    case jamlet_cubic:
      ipredict(sample, n, get_second_update_mask_jamlet_cubic(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_jamlet_cubic(), level, 1, false);
      ipredict(sample, n, get_first_update_mask_jamlet_cubic(), level, 1, false);
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_jamlet_cubic(), level, 1, 1, false);
      break;
    case jamlet_4_point:
      ipredict(sample, n, get_update_mask_jamlet_4_point(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_jamlet_4_point(), level, 1, false);
      break;
    case cdf_5_3:
      lifting::ipredict(sample, n, lifting::get_update_mask_cdf_5_3(), level, 1, false);
      lifting::iupdate(sample, n, lifting::get_prediction_mask_cdf_5_3(), level, 1, false);
      break;
    case cdf_9_7:
      lifting::iscale_even(sample, n, 1.0 / lifting::get_even_scaling_factor_cdf_9_7(), level, 1, 1, false);
      lifting::iscale_odd(sample, n, 1.0 / lifting::get_odd_scaling_factor_cdf_9_7(), level, 1, 1, false);
      lifting::ipredict(sample, n, lifting::get_second_update_mask_cdf_9_7(), level, 1, false);
      lifting::iupdate(sample, n, lifting::get_second_prediction_mask_cdf_9_7(), level, 1, false);
      lifting::ipredict(sample, n, lifting::get_first_update_mask_cdf_9_7(), level, 1, false);
      lifting::iupdate(sample, n, lifting::get_first_prediction_mask_cdf_9_7(), level, 1, false);
      break;
    case cubic_bsplines:
      iupdate(sample, n, get_prediction_mask_cubic_bsplines(), level, 1, false);
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_cubic_bsplines(), level, 1, 1, false);
      ipredict(sample, n, get_update_mask_cubic_bsplines(), level, 1, false);
      break;
    case cubic_bspline_wavelets:
      ipredict(sample, n, get_second_update_mask_cubic_bspline_wavelets(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_cubic_bspline_wavelets(), level, 1, false);
      ipredict(sample, n, get_first_update_mask_cubic_bspline_wavelets(), level, 1, false);
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_cubic_bspline_wavelets(), level, 1, 1, false);
      break;
    case daubechies_d4:
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_daubechies_d4(), level, 1, 1, false);
      iscale_odd(sample, n, 1.0 / get_odd_scaling_factor_daubechies_d4(), level, 1, 1, false);
      ipredict(sample, n, get_second_update_mask_daubechies_d4(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_daubechies_d4(), level, 1, false);
      ipredict(sample, n, get_first_update_mask_daubechies_d4(), level, 1, false);
      break;
    case haar:
      ipredict(sample, n, get_update_mask_haar(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_haar(), level, 1, false);
      break;
    case four_point:
      ipredict(sample, n, get_update_mask_4_point(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_4_point(), level, 1, false);
      break;
    case chaikin:
      ipredict(sample, n, get_second_update_mask_chaikin(), level, 1, false);
      iupdate(sample, n, get_prediction_mask_chaikin(), level, 1, false);
      iscale_even(sample, n, 1.0 / get_even_scaling_factor_chaikin(), level, 1, 1, false);
      ipredict(sample, n, get_update_mask_chaikin(), level, 1, false);
      break;
    case custom:
      inverse_custom_biorthogonal(sample, n, level, custom_steps);
      break;
    };
  }

void make_scaling_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  m.values.resize(n);
  for (auto& v : m.values)
    v = 0.0;
  m.values[n / 2] = 1.0;
  switch (s)
    {
    case jamlet_linear:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_linear(m.values.data(), n, lev);
      break;
    case jamlet_quadratic:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_quadratic(m.values.data(), n, lev);
      break;
    case jamlet_cubic:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_cubic(m.values.data(), n, lev);
      break;
    case jamlet_4_point:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_4_point(m.values.data(), n, lev);
      break;
    case cdf_5_3:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cdf_5_3(m.values.data(), n, lev);
      break;
    case cdf_9_7:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cdf_9_7(m.values.data(), n, lev);
      break;
    case chaikin:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_chaikin(m.values.data(), n, lev);
      break;
    case cubic_bsplines:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cubic_bsplines(m.values.data(), n, lev);
      break;
    case cubic_bspline_wavelets:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cubic_bspline_wavelets(m.values.data(), n, lev);
      break;
    case daubechies_d4:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_daubechies_d4(m.values.data(), n, lev);
      break;
    case four_point:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_4_point(m.values.data(), n, lev);
      break;
    case haar:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_haar(m.values.data(), n, lev);
      break;
    case custom:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        inverse_custom(m.values.data(), n, lev, custom_steps);
      break;
    };
  }

void make_wavelet_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  m.values.resize(n);
  for (auto& v : m.values)
    v = 0.0;
  m.values[n / 2 + ((uint64_t)1 << (uint64_t)(m.levels - get_width(s)))] = 1.0;
  switch (s)
    {
    case jamlet_linear:

      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_linear(m.values.data(), n, lev);
      break;
    case jamlet_quadratic:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_quadratic(m.values.data(), n, lev);
      break;
    case jamlet_cubic:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_cubic(m.values.data(), n, lev);
      break;
    case jamlet_4_point:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_jamlet_4_point(m.values.data(), n, lev);
      break;
    case cdf_5_3:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cdf_5_3(m.values.data(), n, lev);
      break;
    case cdf_9_7:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cdf_9_7(m.values.data(), n, lev);
      break;
    case chaikin:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_chaikin(m.values.data(), n, lev);
      break;
    case cubic_bsplines:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cubic_bsplines(m.values.data(), n, lev);
      break;
    case cubic_bspline_wavelets:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_cubic_bspline_wavelets(m.values.data(), n, lev);
      break;
    case daubechies_d4:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_daubechies_d4(m.values.data(), n, lev);
      break;
    case four_point:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_4_point(m.values.data(), n, lev);
      break;
    case haar:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        lifting::inverse_haar(m.values.data(), n, lev);
      break;
    case custom:
      for (int lev = m.levels - get_width(s); lev >= 0; --lev)
        inverse_custom(m.values.data(), n, lev, custom_steps);
      break;
    };
  }

void make_biorthogonal_scaling_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  m.values.resize(n);
  for (auto& v : m.values)
    v = 0.0;
  m.values[n / 2] = 1.0;
  for (int lev = m.levels - get_width(s); lev >= 0; --lev)
    biorthogonal_inverse(m.values.data(), n, lev, s, custom_steps);
  }

void make_biorthogonal_wavelet_function(model& m, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  m.values.resize(n);
  for (auto& v : m.values)
    v = 0.0;


  m.values[n / 2 + ((uint64_t)1 << (uint64_t)(m.levels - get_width(s)))] = 1.0;
  for (int lev = m.levels - get_width(s); lev >= 0; --lev)
    biorthogonal_inverse(m.values.data(), n, lev, s, custom_steps);

  }

void fill_render_data(model& m, const std::vector<double>& values)
  {
  m.delete_render_objects();
  double cutoff = 1000000.0;
  double max_value = *std::max_element(m.values.begin(), m.values.end());
  double min_value = *std::min_element(m.values.begin(), m.values.end());
  if (max_value > cutoff)
    max_value = cutoff;
  if (min_value < -cutoff)
    min_value = -cutoff;
  if (min_value == max_value)
    {
    min_value -= 1e-4;
    max_value += 1e-4;
    }
  double range_original = max_value - min_value;

  double max_value2 = *std::max_element(values.begin(), values.end());
  double min_value2 = *std::min_element(values.begin(), values.end());

  double range2 = max_value2 - min_value2;

  if (range2 <= range_original)
    {
    double diff = range_original - range2;
    min_value = min_value2 - diff / 2.0;
    max_value = max_value2 + diff / 2.0;
    }
  else
    {
    double diff = range2 - range_original;
    min_value = min_value2 + diff / 2.0;
    max_value = max_value2 - diff / 2.0;
    }


  std::vector<GLfloat> vertices;
  vertices.reserve(values.size() * 2);
  uint64_t n = values.size();
  for (uint64_t i = 0; i < n; ++i)
    {
    float pos_x = (float)i / (float)(n - 1) * 2.f - 1.f;
    float pos_y = ((float)values[i] - (float)min_value) / ((float)max_value - (float)min_value) * 1.8f - 0.9f;
    vertices.push_back(pos_x);
    vertices.push_back(pos_y);
    }

  m._vao = new vertex_array_object();
  m._vao->create();
  gl_check_error(" _vao->create()");
  m._vao->bind();
  gl_check_error(" _vao->bind()");

  m._vbo_array = new buffer_object(GL_ARRAY_BUFFER);
  m._vbo_array->create();
  gl_check_error("_vbo_array->create()");
  m._vbo_array->bind();
  gl_check_error("_vbo_array->bind()");
  m._vbo_array->set_usage_pattern(GL_STATIC_DRAW);
  m._vbo_array->allocate(vertices.data(), (int)(sizeof(GLfloat) * vertices.size()));
  gl_check_error("_vbo_array->allocate()");
  }

double compute_volume(const std::vector<double>& values)
  {
  double v = 0.0;
  for (const auto& value : values)
    v += value;
  return v;
  }

namespace
  {
  double f1(double x)
    {
    return 0.75 * std::exp(-pow(9.0*x - 2.0, 2.0) / 4.0) +
      0.75 * std::exp(-pow(9.0*x + 1.0, 2.0) / 49.0) +
      0.5 * std::exp(-pow(9.0*x - 7.0, 2.0) / 4.0) +
      -0.2 * std::exp(-pow(9.0*x - 4.0, 2.0));
    }

  double f2(double x)
    {
    return 9.0*x*(std::exp(-1.0 / 9.0*pow(pow(x, 2.0), 1.0 / 4.0)) - 1.0);
    }

  double f3(double x)
    {
    return 97.0*(x - 0.5)*std::tanh(1.0 / 97.0*pow(pow(x - 0.5, 2.0), 1.0 / 4.0));
    }

  double f4(double x)
    {
    return std::exp(-std::abs(x));
    }

  double f5(double x)
    {
    return pow(pow(2.0*x - 1.0, 2.0), 0.25);
    }

  inline double fract(double f)
    {
    return f - std::floor(f);
    }

  double f6(double x)
    {
    double noise = (fract(std::sin(x)*753.5453123f)*2.0 - 1.0)*1e-2;
    return f1(x) + noise;
    }

  double f7(double x)
    {
    double noise = (fract(std::sin(x)*753.5453123f)*2.0 - 1.0)*1e-2;
    return f2(x) + noise;
    }

  double f8(double x)
    {
    double noise = (fract(std::sin(x)*753.5453123f)*2.0 - 1.0)*1e-2;
    return f3(x) + noise;
    }

  double f9(double x)
    {
    double noise = (fract(std::sin(x)*753.5453123f)*2.0 - 1.0)*1e-2;
    return f4(x) + noise;
    }

  double f10(double x)
    {
    double noise = (fract(std::sin(x)*753.5453123f)*2.0 - 1.0)*1e-2;
    return f5(x) + noise;
    }
  }

void make_test_function(model& m, int f)
  {
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  m.values.resize(n);
  for (auto& v : m.values)
    v = 0.0;

  for (uint64_t i = 0; i < m.values.size(); ++i)
    {
    double x = (double)i / (double)(n - 1) * 2.f - 1.f;
    switch (f)
      {
      case 0: m.values[i] = f1(x); break;
      case 1: m.values[i] = f2(x); break;
      case 2: m.values[i] = f3(x); break;
      case 3: m.values[i] = f4(x); break;
      case 4: m.values[i] = f5(x); break;
      case 5: m.values[i] = f6(x); break;
      case 6: m.values[i] = f7(x); break;
      case 7: m.values[i] = f8(x); break;
      case 8: m.values[i] = f9(x); break;
      case 9: m.values[i] = f10(x); break;
      }
    }
  }

void get_spline_component(std::vector<double>& values, const model& m, int _level, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  values = m.values;
  int lifting_steps = m.levels - _level;
  for (int lev = 0; lev < lifting_steps; ++lev)
    {
    switch (s)
      {
      case jamlet_linear: forward_jamlet_linear(values.data(), n, lev); break;
      case jamlet_quadratic: forward_jamlet_quadratic(values.data(), n, lev); break;
      case jamlet_cubic: forward_jamlet_cubic(values.data(), n, lev); break;
      case jamlet_4_point: forward_jamlet_4_point(values.data(), n, lev); break;
      case cdf_5_3: forward_cdf_5_3(values.data(), n, lev); break;
      case cdf_9_7: forward_cdf_9_7(values.data(), n, lev); break;
      case chaikin: forward_chaikin(values.data(), n, lev); break;
      case cubic_bsplines: forward_cubic_bsplines(values.data(), n, lev); break;
      case cubic_bspline_wavelets: forward_cubic_bspline_wavelets(values.data(), n, lev); break;
      case daubechies_d4: forward_daubechies_d4(values.data(), n, lev); break;
      case four_point: forward_4_point(values.data(), n, lev); break;
      case haar: forward_haar(values.data(), n, lev); break;
      case custom: forward_custom(values.data(), n, lev, custom_steps); break;
      }
    }
  compress(values.data(), n, std::numeric_limits<double>::infinity(), lifting_steps);
  for (int lev = lifting_steps - 1; lev >= 0; --lev)
    {
    switch (s)
      {
      case jamlet_linear: inverse_jamlet_linear(values.data(), n, lev); break;
      case jamlet_quadratic: inverse_jamlet_quadratic(values.data(), n, lev); break;
      case jamlet_cubic: inverse_jamlet_cubic(values.data(), n, lev); break;
      case jamlet_4_point: inverse_jamlet_4_point(values.data(), n, lev); break;
      case cdf_5_3: inverse_cdf_5_3(values.data(), n, lev); break;
      case cdf_9_7: inverse_cdf_9_7(values.data(), n, lev); break;
      case chaikin: inverse_chaikin(values.data(), n, lev); break;
      case cubic_bsplines: inverse_cubic_bsplines(values.data(), n, lev); break;
      case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(values.data(), n, lev); break;
      case daubechies_d4: inverse_daubechies_d4(values.data(), n, lev); break;
      case four_point: inverse_4_point(values.data(), n, lev); break;
      case haar: inverse_haar(values.data(), n, lev); break;
      case custom: inverse_custom(values.data(), n, lev, custom_steps); break;
      }
    }
  }

void get_wavelet_component(std::vector<double>& values, const model& m, int _level, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  values = m.values;
  int lifting_steps = m.levels - _level;
  for (int lev = 0; lev < lifting_steps; ++lev)
    {
    switch (s)
      {
      case jamlet_linear: forward_jamlet_linear(values.data(), n, lev); break;
      case jamlet_quadratic: forward_jamlet_quadratic(values.data(), n, lev); break;
      case jamlet_cubic: forward_jamlet_cubic(values.data(), n, lev); break;
      case jamlet_4_point: forward_jamlet_4_point(values.data(), n, lev); break;
      case cdf_5_3: forward_cdf_5_3(values.data(), n, lev); break;
      case cdf_9_7: forward_cdf_9_7(values.data(), n, lev); break;
      case chaikin: forward_chaikin(values.data(), n, lev); break;
      case cubic_bsplines: forward_cubic_bsplines(values.data(), n, lev); break;
      case cubic_bspline_wavelets: forward_cubic_bspline_wavelets(values.data(), n, lev); break;
      case daubechies_d4: forward_daubechies_d4(values.data(), n, lev); break;
      case four_point: forward_4_point(values.data(), n, lev); break;
      case haar: forward_haar(values.data(), n, lev); break;
      case custom: forward_custom(values.data(), n, lev, custom_steps); break;
      }
    }

  const uint64_t mask = ((uint64_t)1 << (lifting_steps)) - 1;
  const uint64_t mask2 = ((uint64_t)1 << (lifting_steps - 1)) - 1;
  for (uint64_t i = 0; i < n; ++i)
    {
    if ((i & mask) == 0) // spline space
      values[i] = 0.0;
    else
      {
      if ((i & mask2) != 0) // not in wavelet space
        values[i] = 0.0;
      }

    }

  for (int lev = lifting_steps - 1; lev >= 0; --lev)
    {
    switch (s)
      {
      case jamlet_linear: inverse_jamlet_linear(values.data(), n, lev); break;
      case jamlet_quadratic: inverse_jamlet_quadratic(values.data(), n, lev); break;
      case jamlet_cubic: inverse_jamlet_cubic(values.data(), n, lev); break;
      case jamlet_4_point: inverse_jamlet_4_point(values.data(), n, lev); break;
      case cdf_5_3: inverse_cdf_5_3(values.data(), n, lev); break;
      case cdf_9_7: inverse_cdf_9_7(values.data(), n, lev); break;
      case chaikin: inverse_chaikin(values.data(), n, lev); break;
      case cubic_bsplines: inverse_cubic_bsplines(values.data(), n, lev); break;
      case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(values.data(), n, lev); break;
      case daubechies_d4: inverse_daubechies_d4(values.data(), n, lev); break;
      case four_point: inverse_4_point(values.data(), n, lev); break;
      case haar: inverse_haar(values.data(), n, lev); break;
      case custom: inverse_custom(values.data(), n, lev, custom_steps); break;
      }
    }
  }

double compress(model& m, double threshold, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);

  for (int lev = 0; lev < m.levels; ++lev)
    {
    switch (s)
      {
      case jamlet_linear: forward_jamlet_linear(m.values.data(), n, lev); break;
      case jamlet_quadratic: forward_jamlet_quadratic(m.values.data(), n, lev); break;
      case jamlet_cubic: forward_jamlet_cubic(m.values.data(), n, lev); break;
      case jamlet_4_point: forward_jamlet_4_point(m.values.data(), n, lev); break;
      case cdf_5_3: forward_cdf_5_3(m.values.data(), n, lev); break;
      case cdf_9_7: forward_cdf_9_7(m.values.data(), n, lev); break;
      case chaikin: forward_chaikin(m.values.data(), n, lev); break;
      case cubic_bsplines: forward_cubic_bsplines(m.values.data(), n, lev); break;
      case cubic_bspline_wavelets: forward_cubic_bspline_wavelets(m.values.data(), n, lev); break;
      case daubechies_d4: forward_daubechies_d4(m.values.data(), n, lev); break;
      case four_point: forward_4_point(m.values.data(), n, lev); break;
      case haar: forward_haar(m.values.data(), n, lev); break;
      case custom: forward_custom(m.values.data(), n, lev, custom_steps); break;
      }
    }
  uint64_t compressed = compress(m.values.data(), n, threshold, m.levels);

  for (int lev = m.levels - 1; lev >= 0; --lev)
    {
    switch (s)
      {
      case jamlet_linear: inverse_jamlet_linear(m.values.data(), n, lev); break;
      case jamlet_quadratic: inverse_jamlet_quadratic(m.values.data(), n, lev); break;
      case jamlet_cubic: inverse_jamlet_cubic(m.values.data(), n, lev); break;
      case jamlet_4_point: inverse_jamlet_4_point(m.values.data(), n, lev); break;
      case cdf_5_3: inverse_cdf_5_3(m.values.data(), n, lev); break;
      case cdf_9_7: inverse_cdf_9_7(m.values.data(), n, lev); break;
      case chaikin: inverse_chaikin(m.values.data(), n, lev); break;
      case cubic_bsplines: inverse_cubic_bsplines(m.values.data(), n, lev); break;
      case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(m.values.data(), n, lev); break;
      case daubechies_d4: inverse_daubechies_d4(m.values.data(), n, lev); break;
      case four_point: inverse_4_point(m.values.data(), n, lev); break;
      case haar: inverse_haar(m.values.data(), n, lev); break;
      case custom: inverse_custom(m.values.data(), n, lev, custom_steps); break;
      }
    }
  return (double)compressed / (double)n;
  }

void smooth(model& m, double threshold, int smooth_level, scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = ((uint64_t)1 << (uint64_t)m.levels);
  for (int lev = 0; lev < smooth_level; ++lev)
    {
    switch (s)
      {
      case jamlet_linear: forward_jamlet_linear(m.values.data(), n, lev); break;
      case jamlet_quadratic: forward_jamlet_quadratic(m.values.data(), n, lev); break;
      case jamlet_cubic: forward_jamlet_cubic(m.values.data(), n, lev); break;
      case jamlet_4_point: forward_jamlet_4_point(m.values.data(), n, lev); break;
      case cdf_5_3: forward_cdf_5_3(m.values.data(), n, lev); break;
      case cdf_9_7: forward_cdf_9_7(m.values.data(), n, lev); break;
      case chaikin: forward_chaikin(m.values.data(), n, lev); break;
      case cubic_bsplines: forward_cubic_bsplines(m.values.data(), n, lev); break;
      case cubic_bspline_wavelets: forward_cubic_bspline_wavelets(m.values.data(), n, lev); break;
      case daubechies_d4: forward_daubechies_d4(m.values.data(), n, lev); break;
      case four_point: forward_4_point(m.values.data(), n, lev); break;
      case haar: forward_haar(m.values.data(), n, lev); break;
      case custom: forward_custom(m.values.data(), n, lev, custom_steps); break;
      }
    }

  smooth(m.values.data(), n, threshold, smooth_level);

  for (int lev = smooth_level - 1; lev >= 0; --lev)
    {
    switch (s)
      {
      case jamlet_linear: inverse_jamlet_linear(m.values.data(), n, lev); break;
      case jamlet_quadratic: inverse_jamlet_quadratic(m.values.data(), n, lev); break;
      case jamlet_cubic: inverse_jamlet_cubic(m.values.data(), n, lev); break;
      case jamlet_4_point: inverse_jamlet_4_point(m.values.data(), n, lev); break;
      case cdf_5_3: inverse_cdf_5_3(m.values.data(), n, lev); break;
      case cdf_9_7: inverse_cdf_9_7(m.values.data(), n, lev); break;
      case chaikin: inverse_chaikin(m.values.data(), n, lev); break;
      case cubic_bsplines: inverse_cubic_bsplines(m.values.data(), n, lev); break;
      case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(m.values.data(), n, lev); break;
      case daubechies_d4: inverse_daubechies_d4(m.values.data(), n, lev); break;
      case four_point: inverse_4_point(m.values.data(), n, lev); break;
      case haar: inverse_haar(m.values.data(), n, lev); break;
      case custom: inverse_custom(m.values.data(), n, lev, custom_steps); break;
      }
    }
  }

std::vector<lifting_step> parse(const std::string& wavelet_rules)
  {
  std::vector<lifting_step> rules;
  try
    {
    auto tokes = tokenize(wavelet_rules);
    Program prog = make_program(tokes);
    for (const auto& stm : prog.statements)
      {
      if (std::holds_alternative<Tag>(stm))
        {
        Tag t = std::get<Tag>(stm);
        if (t.name == "predict")
          {
          rules.emplace_back();
          rules.back().type = lst_predict;
          }
        else if (t.name == "update")
          {
          rules.emplace_back();
          rules.back().type = lst_update;
          }
        else if (t.name == "scale_even")
          {
          rules.emplace_back();
          rules.back().type = lst_scale_even;
          }
        else if (t.name == "scale_odd")
          {
          rules.emplace_back();
          rules.back().type = lst_scale_odd;
          }
        }
      else
        {
        if (rules.empty())
          throw std::logic_error("error: tag missing (valid tags are update, predict, scale_even, scale_odd)");
        Expression e = std::get<Expression>(stm);
        double v = get_value(e);
        rules.back().mask.push_back(v);
        }
      }
    }
  catch (std::logic_error e)
    {
    Logging::Error() << e.what() << "\n";
    return rules;
    }
  Logging::Info() << "Compile succeeded\n";
  analyze((scheme)custom, rules);
  return rules;
  }

double compute_smoothness(const std::vector<double>& samples, double scale)
  {
  std::vector<double> P = samples;
  auto it = P.begin();
  while (it != P.end() && !(*it))
    ++it;
  P.erase(P.begin(), it);
  while (!P.empty() && !P.back())
    P.pop_back();
  for (auto& p : P)
    p *= scale;
  double sob = std::numeric_limits<double>::quiet_NaN();
  try
    {
    sob = sobsmthest(P);
    }
  catch (std::logic_error e)
    {
    Logging::Error() << e.what() << "\n";
    }
  return sob;
  }

void analyze(scheme s, const std::vector<lifting_step>& custom_steps)
  {
  using namespace lifting;
  uint64_t n = 32;
  std::vector<double> samples((size_t)n, 0.0);
  samples[n / 2] = 1.0;
  switch (s)
    {
    case jamlet_linear: inverse_jamlet_linear(samples.data(), n, 0); break;
    case jamlet_quadratic: inverse_jamlet_quadratic(samples.data(), n, 0); break;
    case jamlet_cubic: inverse_jamlet_cubic(samples.data(), n, 0); break;
    case jamlet_4_point: inverse_jamlet_4_point(samples.data(), n, 0); break;
    case cdf_5_3: inverse_cdf_5_3(samples.data(), n, 0); break;
    case cdf_9_7: inverse_cdf_9_7(samples.data(), n, 0); break;
    case chaikin: inverse_chaikin(samples.data(), n, 0); break;
    case cubic_bsplines: inverse_cubic_bsplines(samples.data(), n, 0); break;
    case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(samples.data(), n, 0); break;
    case daubechies_d4: inverse_daubechies_d4(samples.data(), n, 0); break;
    case four_point: inverse_4_point(samples.data(), n, 0); break;
    case haar: inverse_haar(samples.data(), n, 0); break;
    case custom: inverse_custom(samples.data(), n, 0, custom_steps); break;
    }
  double sob_scaling = compute_smoothness(samples);
  Logging::GetInstance() << "Scaling coeff: ";
  for (uint64_t i = 0; i < n; ++i)
    Logging::GetInstance() << samples[i] << " ";
  Logging::GetInstance() << "\n";
  double scaling_sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  if (scaling_sum != 2.0)
    {
    Logging::GetInstance() << "Sum of scaling coefficients = " << scaling_sum << "\n";
    Logging::GetInstance() << "It is advisable to add a even scale step with value " << scaling_sum / 2.0 << "\n";
    }
  for (auto& smpl : samples)
    smpl = 0.0;
  samples[n / 2] = 1.0;
  biorthogonal_inverse(samples.data(), n, 0, s, custom_steps);
  double sob_scaling_dual = compute_smoothness(samples, 2.0);
  Logging::GetInstance() << "Biorthogonal scaling coeff: ";
  for (uint64_t i = 0; i < n; ++i)
    Logging::GetInstance() << samples[i] << " ";
  Logging::GetInstance() << "\n";
  for (auto& smpl : samples)
    smpl = 0.0;
  samples[n / 2 + 1] = 1.0;
  switch (s)
    {
    case jamlet_linear: inverse_jamlet_linear(samples.data(), n, 0); break;
    case jamlet_quadratic: inverse_jamlet_quadratic(samples.data(), n, 0); break;
    case jamlet_cubic: inverse_jamlet_cubic(samples.data(), n, 0); break;
    case jamlet_4_point: inverse_jamlet_4_point(samples.data(), n, 0); break;
    case cdf_5_3: inverse_cdf_5_3(samples.data(), n, 0); break;
    case cdf_9_7: inverse_cdf_9_7(samples.data(), n, 0); break;
    case chaikin: inverse_chaikin(samples.data(), n, 0); break;
    case cubic_bsplines: inverse_cubic_bsplines(samples.data(), n, 0); break;
    case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(samples.data(), n, 0); break;
    case daubechies_d4: inverse_daubechies_d4(samples.data(), n, 0); break;
    case four_point: inverse_4_point(samples.data(), n, 0); break;
    case haar: inverse_haar(samples.data(), n, 0); break;
    case custom: inverse_custom(samples.data(), n, 0, custom_steps); break;
    }
  Logging::GetInstance() << "Wavelet coeff: ";
  for (uint64_t i = 0; i < n; ++i)
    Logging::GetInstance() << samples[i] << " ";
  Logging::GetInstance() << "\n";
  double current_sum = std::accumulate(samples.begin(), samples.end(), 0.0);

  std::vector<double> sample_vm((size_t)n, 0.0);
  sample_vm[n / 2 + 1] = 1.0;
  std::vector<double> vanishing_moment((size_t)2, 1.0);
  iupdate(sample_vm.data(), n, vanishing_moment, 0, 1, false);
  switch (s)
    {
    case jamlet_linear: inverse_jamlet_linear(sample_vm.data(), n, 0); break;
    case jamlet_quadratic: inverse_jamlet_quadratic(sample_vm.data(), n, 0); break;
    case jamlet_cubic: inverse_jamlet_cubic(sample_vm.data(), n, 0); break;
    case jamlet_4_point: inverse_jamlet_4_point(sample_vm.data(), n, 0); break;
    case cdf_5_3: inverse_cdf_5_3(sample_vm.data(), n, 0); break;
    case cdf_9_7: inverse_cdf_9_7(sample_vm.data(), n, 0); break;
    case chaikin: inverse_chaikin(sample_vm.data(), n, 0); break;
    case cubic_bsplines: inverse_cubic_bsplines(sample_vm.data(), n, 0); break;
    case cubic_bspline_wavelets: inverse_cubic_bspline_wavelets(sample_vm.data(), n, 0); break;
    case daubechies_d4: inverse_daubechies_d4(sample_vm.data(), n, 0); break;
    case four_point: inverse_4_point(sample_vm.data(), n, 0); break;
    case haar: inverse_haar(sample_vm.data(), n, 0); break;
    case custom: inverse_custom(sample_vm.data(), n, 0, custom_steps); break;
    }
  double after_update_sum = std::accumulate(sample_vm.begin(), sample_vm.end(), 0.0);
  double update_mask_value = -current_sum / (after_update_sum - current_sum);
  Logging::GetInstance() << "Current wavelet sum is " << current_sum << "\n";
  if (current_sum)
    {
    Logging::GetInstance() << "Add update step with mask value " << update_mask_value << " for one vanishing moment\n";
    }
  Logging::GetInstance() << "Riesz basis for ]" << -sob_scaling_dual << ", " << sob_scaling << "[\n";
  }


namespace
  {
  double compute_vanishing_moment(const std::vector<lifting_step>& custom_steps)
    {
    using namespace lifting;
    uint64_t n = 32;
    std::vector<double> samples((size_t)n, 0.0);
    samples[n / 2 + 1] = 1.0;
    inverse_custom(samples.data(), n, 0, custom_steps);
    double current_sum = std::accumulate(samples.begin(), samples.end(), 0.0);

    std::vector<double> sample_vm((size_t)n, 0.0);
    sample_vm[n / 2 + 1] = 1.0;
    std::vector<double> vanishing_moment((size_t)2, 1.0);
    iupdate(sample_vm.data(), n, vanishing_moment, 0, 1, false);
    inverse_custom(sample_vm.data(), n, 0, custom_steps);
    double after_update_sum = std::accumulate(sample_vm.begin(), sample_vm.end(), 0.0);
    double update_mask_value = -current_sum / (after_update_sum - current_sum);
    return update_mask_value;
    }

  void _compute_smoothness(const std::vector<lifting_step>& custom_steps, double& sob, double& sob_dual)
    {
    using namespace lifting;
    uint64_t n = 32;
    std::vector<double> samples((size_t)n, 0.0);
    samples[n / 2] = 1.0;
    inverse_custom(samples.data(), n, 0, custom_steps);
    sob = compute_smoothness(samples);
    for (auto& smpl : samples)
      smpl = 0.0;
    samples[n / 2] = 1.0;
    biorthogonal_inverse(samples.data(), n, 0, custom, custom_steps);
    sob_dual = compute_smoothness(samples, 2.0);
    }

  void do_construction(double alpha, std::vector<lifting_step>& custom_steps, double& sob, double& sob_dual)
    {
    assert(!custom_steps.empty());
    assert(custom_steps.back().type == lst_update);
    custom_steps.back().mask[0] = alpha;
    custom_steps.back().mask[1] = 0.0;
    custom_steps.back().mask[2] = 0.0;
    custom_steps.back().mask[3] = alpha;
    double vm = compute_vanishing_moment(custom_steps);
    custom_steps.back().mask[1] = vm;
    custom_steps.back().mask[2] = vm;
    _compute_smoothness(custom_steps, sob, sob_dual);
    }
  }

void construct_stable_wavelet(std::vector<lifting_step>& custom_steps, double& sob, double& sob_dual)
  {
  custom_steps.emplace_back();
  custom_steps.back().type = lst_update;
  custom_steps.back().mask.resize(4);
  sob_dual = -1000000000000.0;
  double max_coeff = 2.0;
  double alpha = -max_coeff;
  double best_alpha = -max_coeff;
  double step_size = 0.0001;
  while (alpha <= max_coeff)
    {
    double current_sob_dual;
    do_construction(alpha, custom_steps, sob, current_sob_dual);
    if (current_sob_dual == current_sob_dual && sob == sob)
      {
      if (current_sob_dual > sob_dual)
        {
        sob_dual = current_sob_dual;
        best_alpha = alpha;
        }
      }
    alpha += step_size;
    }
  do_construction(best_alpha, custom_steps, sob, sob_dual);
  }