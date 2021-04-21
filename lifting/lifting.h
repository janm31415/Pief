///////////////////////////////////////////////////////////////////////////////
//
// Lifting   :  Implementation of the lifting scheme
//
// Author    :  Jan Maes                                            
// Version   :  1.1
// Date      :  23 January 2019
// License   :  MIT License
//
///////////////////////////////////////////////////////////////////////////////

/*
Changelog:
V1.0: 27 September 2019
  - first version
V1.1: 23 January 2020
  - added strides to buffers
  - added support for cyclical buffers
*/


#pragma once

#include <vector>
#include <cassert>
#include <stdint.h>

namespace lifting
  {

  inline bool is_multiple_of_power_of_two(uint64_t n, uint64_t level)
    {
    uint64_t pow_two = (uint64_t)1 << level;
    return (n & (pow_two - 1)) == 0;
    }

  template <typename T>
  void scale_even(T* sample, uint64_t n, double s, uint64_t level, int64_t only_scale_away_from_border, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const int64_t offset = (int64_t)((uint64_t)1 << (level + 1));
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)n ; i += offset)
        sample[i*stride] = (T)(sample[i*stride] * s);
      }
    else
      {
      for (int64_t i = (int64_t)only_scale_away_from_border*offset; i < (int64_t)n - (int64_t)only_scale_away_from_border*offset; i += offset)
        sample[i*stride] = (T)(sample[i*stride] * s);
      }
    }

  template <typename T>
  void scale_odd(T* sample, uint64_t n, double s, uint64_t level, int64_t only_scale_away_from_border, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const int64_t offset = (int64_t)((uint64_t)1 << (level + 1));
    if (cyclical)
      {
      for (int64_t i = (int64_t)(((uint64_t)1 << level)); i < (int64_t)n; i += offset)
        sample[i*stride] = (T)(sample[i*stride] * s);
      }
    else
      {
      for (int64_t i = (int64_t)(((uint64_t)1 << level) + only_scale_away_from_border * offset); i < (int64_t)n - (int64_t)only_scale_away_from_border*offset; i += offset)
        sample[i*stride] = (T)(sample[i*stride] * s);
      }
    }

  /*
  The predict stencil is defined by double mask.
  The left and right odd points are predicted with the same stencil value (mask)

  template <typename T>
  void predict(T* sample, uint64_t n, double mask, uint64_t level, uint64_t stride)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    for (int64_t i = 0; i < (int64_t)(max_i-1); ++i)
      {
      const double prediction = mask * (sample[((uint64_t)i << (level + 1))*stride] + sample[((uint64_t)(i + 1) << (level + 1))*stride]);
      sample[(((i << 1) + 1) << level)*stride] -= (T)prediction;
      }
    sample[(((max_i << 1) - 1) << level)*stride] -= (T)(2.0*mask*sample[((uint64_t)(max_i - 1) << (level + 1))*stride]);
    }
    */
  template <typename T>
  void predict(T* sample, uint64_t n, const std::vector<double>& mask, uint64_t level, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    const int64_t offset = -(int64_t)(mask.size() >> 1) + 1;    
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double prediction = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (i + j + offset) << (int64_t)(level + 1);
          if (coeff < 0)
            coeff += (int64_t)n;
          else if (coeff >= (int64_t)n)
            coeff -= (int64_t)n;
          prediction += mask[j] * sample[coeff*stride];
          }
        sample[(((i << 1) + 1) << level)*stride] -= (T)prediction;
        }
      }
    else
      {
      const int64_t max_coeff = (max_i - 1) << (int64_t)(level + 1);
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double prediction = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (i + j + offset) << (int64_t)(level + 1);
          if (coeff < 0)
            coeff = 0;
          else if (coeff >= (int64_t)n)
            coeff = max_coeff;
          prediction += mask[j] * sample[coeff*stride];
          }
        sample[(((i << 1) + 1) << level)*stride] -= (T)prediction;
        }
      }
    }

  /*
  The update stencil is defined by double mask.
  The left and right even points are updated with the same stencil value (mask)

  template <typename T>
  void update(T* sample, uint64_t n, double mask, uint64_t level, uint64_t stride)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    sample[0] += (T)(2.0*mask*sample[((uint64_t)1 << (level))*stride]);
    for (int64_t i = 1; i < (int64_t)(max_i); ++i)
      {
      const double upd = mask * (sample[((uint64_t)(i * 2 - 1) << level)*stride] + sample[((uint64_t)(i * 2 + 1) << (level))*stride]);
      sample[(i << (level + 1))*stride] += (T)upd;
      }
    }
    */
  template <typename T>
  void update(T* sample, uint64_t n, const std::vector<double>& mask, uint64_t level, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = (n >> (level + 1));
    const int64_t offset = -(int64_t)(mask.size() >> 1) + 1;
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double upd = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (((i + j + offset) << 1) - 1) << (int64_t)(level);
          if (coeff < 0)
            coeff += (int64_t)n;
          else if (coeff >= (int64_t)n)
            coeff -= (int64_t)n;
          upd += mask[j] * sample[coeff*stride];
          }
        sample[(i << (level + 1))*stride] += (T)upd;
        }
      }
    else
      {
      const int64_t min_coeff = (int64_t)1 << (int64_t)(level);
      const int64_t max_coeff = ((max_i << 1) - 1) << (int64_t)(level);
      for (int64_t i = 1; i < (int64_t)max_i; ++i)
        {
        double upd = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (((i + j + offset) << 1) - 1) << (int64_t)(level);
          if (coeff < 0)
            coeff = min_coeff;
          else if (coeff >= (int64_t)n)
            coeff = max_coeff;
          upd += mask[j] * sample[coeff*stride];
          }
        sample[(i << (level + 1))*stride] += (T)upd;
        }
      }
    }

  template <typename T>
  void iscale_even(T* sample, uint64_t n, double s, uint64_t level, int64_t only_scale_away_from_border, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const int64_t offset = (int64_t)((uint64_t)1 << (level + 1));
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)n; i += offset)
        sample[i*stride] = (T)(sample[i*stride] / s);
      }
    else
      {
      for (int64_t i = (int64_t)only_scale_away_from_border*offset; i < (int64_t)n - (int64_t)only_scale_away_from_border*offset; i += offset)
        sample[i*stride] = (T)(sample[i*stride] / s);
      }
    }

  template <typename T>
  void iscale_odd(T* sample, uint64_t n, double s, uint64_t level, int64_t only_scale_away_from_border, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const int64_t offset = (int64_t)((uint64_t)1 << (level + 1));
    if (cyclical)
      {
      for (int64_t i = (int64_t)(((uint64_t)1 << level)); i < (int64_t)n; i += offset)
        sample[i*stride] = (T)(sample[i*stride] / s);
      }
    else
      {
      for (int64_t i = (int64_t)(((uint64_t)1 << level) + only_scale_away_from_border * offset); i < (int64_t)n - (int64_t)only_scale_away_from_border*offset; i += offset)
        sample[i*stride] = (T)(sample[i*stride] / s);
      }
    }

  /*
  The predict stencil is defined by double mask.
  The left and right odd points are predicted with the same stencil value (mask)

  template <typename T>
  void ipredict(T* sample, uint64_t n, double mask, uint64_t level, uint64_t stride)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    for (int64_t i = 0; i < (int64_t)(max_i-1); ++i)
      {
      const double prediction = mask * (sample[((uint64_t)i << (level + 1))*stride] + sample[((uint64_t)(i + 1) << (level + 1))*stride]);
      sample[(((i << 1) + 1) << level)*stride] += (T)prediction;
      }
    sample[(((max_i << 1) - 1) << level)*stride] += (T)(2.0*mask*sample[((uint64_t)(max_i - 1) << (level + 1))*stride]);
    }
    */
  template <typename T>
  void ipredict(T* sample, uint64_t n, const std::vector<double>& mask, uint64_t level, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    const int64_t offset = -(int64_t)(mask.size() >> 1) + 1;
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double prediction = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (i + j + offset) << (int64_t)(level + 1);
          if (coeff < 0)
            coeff += n;
          else if (coeff >= (int64_t)n)
            coeff -= n;
          prediction += mask[j] * sample[coeff*stride];
          }
        sample[(((i << 1) + 1) << level)*stride] += (T)prediction;
        }
      }
    else
      {
      const int64_t max_coeff = (max_i - 1) << (int64_t)(level + 1);
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double prediction = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (i + j + offset) << (int64_t)(level + 1);
          if (coeff < 0)
            coeff = 0;
          else if (coeff >= (int64_t)n)
            coeff = max_coeff;
          prediction += mask[j] * sample[coeff*stride];
          }
        sample[(((i << 1) + 1) << level)*stride] += (T)prediction;
        }
      }
    }

  /*
   The update stencil is defined by double mask.
   The left and right even points are updated with the same stencil value (mask)

  template <typename T>
  void iupdate(T* sample, uint64_t n, double mask, uint64_t level, uint64_t stride)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = n >> (level + 1);
    sample[0] -= (T)(2.0*mask*sample[((uint64_t)1 << (level))*stride]);
    for (int64_t i = 1; i < (int64_t)(max_i); ++i)
      {
      const double upd = mask * (sample[((uint64_t)(i * 2 - 1) << level)*stride] + sample[((uint64_t)(i * 2 + 1) << (level))*stride]);
      sample[(i << (level + 1))*stride] -= (T)upd;
      }
    }
    */
  template <typename T>
  void iupdate(T* sample, uint64_t n, const std::vector<double>& mask, uint64_t level, uint64_t stride, bool cyclical)
    {
    assert(is_multiple_of_power_of_two(n, level));
    const uint64_t max_i = (n >> (level + 1));
    const int64_t offset = -(int64_t)(mask.size() >> 1) + 1;
    if (cyclical)
      {
      for (int64_t i = 0; i < (int64_t)max_i; ++i)
        {
        double upd = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (((i + j + offset) << 1) - 1) << (int64_t)(level);
          if (coeff < 0)
            coeff += n;
          else if (coeff >= (int64_t)n)
            coeff -= n;
          upd += mask[j] * sample[coeff*stride];
          }
        sample[(i << (level + 1))*stride] -= (T)upd;
        }
      }
    else
      {
      const int64_t min_coeff = (int64_t)1 << (int64_t)(level);
      const int64_t max_coeff = ((max_i << 1) - 1) << (int64_t)(level);
      for (int64_t i = 1; i < (int64_t)max_i; ++i)
        {
        double upd = 0.0;
        for (int64_t j = 0; j < (int64_t)mask.size(); ++j)
          {
          int64_t coeff = (((i + j + offset) << 1) - 1) << (int64_t)(level);
          if (coeff < 0)
            coeff = min_coeff;
          else if (coeff >= (int64_t)n)
            coeff = max_coeff;
          upd += mask[j] * sample[coeff*stride];
          }
        sample[(i << (level + 1))*stride] -= (T)upd;
        }
      }
    }

  /*
  This method assumes that 'multiresolution_levels' forward lifting steps have been computed already.
  */
  template <typename T>
  uint64_t compress(T* sample, uint64_t n, T threshold, uint64_t multiresolution_levels, uint64_t stride = 1)
    {
    uint64_t compressed = 0;
    const uint64_t mask = ((uint64_t)1 << (multiresolution_levels)) - 1;
    for (uint64_t i = 0; i < n; ++i)
      {
      if (i & mask)
        {
        if (std::abs(sample[i*stride]) < threshold)
          {
          sample[i*stride] = (T)0.0;
          ++compressed;
          }
        }
      }
    return compressed;
    }

  /*
  This method assumes that 'multiresolution_levels' forward lifting steps have been computed already.
  */
  template <typename T>
  void smooth(T* sample, uint64_t n, T threshold, uint64_t multiresolution_levels, uint64_t stride = 1)
    {
    const uint64_t mask = ((uint64_t)1 << (multiresolution_levels)) - 1;
    for (uint64_t i = 0; i < n; ++i)
      {
      if (i & mask)
        {
        if (sample[i*stride] > threshold)
          sample[i*stride] -= threshold;
        else if (sample[i*stride] < -threshold)
          sample[i*stride] += threshold;
        else
          sample[i*stride] = (T)0.0;
        }
      }
    }

  inline std::vector<double> compute_symmetric_mask(double value)
    {
    std::vector<double> v;
    v.push_back(value);
    v.push_back(value);
    return v;
    }

  inline std::vector<double> compute_prediction_mask_chaikin()
    {
    std::vector<double> prediction_mask;
    prediction_mask.reserve(2);
    prediction_mask.push_back(0.25);
    prediction_mask.push_back(0.75);
    return prediction_mask;
    }

  inline const std::vector<double>& get_prediction_mask_chaikin()
    {
    static std::vector<double> prediction_mask(compute_prediction_mask_chaikin());
    return prediction_mask;
    }

  inline std::vector<double> compute_update_mask_chaikin()
    {
    std::vector<double> update_mask;
    update_mask.reserve(2);
    update_mask.push_back(0.0);
    update_mask.push_back(-1.0 / 3.0);
    return update_mask;
    }

  inline const std::vector<double>& get_update_mask_chaikin()
    {
    static std::vector<double> update_mask(compute_update_mask_chaikin());
    return update_mask;
    }

  inline const std::vector<double>& get_second_update_mask_chaikin()
    {
    //return 1.0 / 3.0;
    static std::vector<double> mask = compute_symmetric_mask(1.0 / 3.0);
    return mask;
    }

  inline double get_even_scaling_factor_chaikin()
    {
    return 3.0 / 2.0;
    }

  template <typename T>
  void forward_chaikin(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    update(sample, n, get_update_mask_chaikin(), level, stride, cyclical);
    scale_even(sample, n, get_even_scaling_factor_chaikin(), level, 1, stride, cyclical);
    predict(sample, n, get_prediction_mask_chaikin(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_chaikin(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_chaikin(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_second_update_mask_chaikin(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_chaikin(), level, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_chaikin(), level, 1, stride, cyclical);
    iupdate(sample, n, get_update_mask_chaikin(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_cubic_bspline_wavelets()
    {
    static std::vector<double> mask = compute_symmetric_mask(0.5);
    return mask;
    //return 0.5;
    }

  inline const std::vector<double>& get_first_update_mask_cubic_bspline_wavelets()
    {
    //return -0.5;
    static std::vector<double> mask = compute_symmetric_mask(-0.5);
    return mask;
    }

  inline const std::vector<double>& get_second_update_mask_cubic_bspline_wavelets()
    {
    //return 3.0 / 8.0;
    static std::vector<double> mask = compute_symmetric_mask(3.0 / 8.0);
    return mask;
    }

  inline double get_even_scaling_factor_cubic_bspline_wavelets()
    {
    return 2.0;
    }

  template <typename T>
  void forward_cubic_bspline_wavelets(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    scale_even(sample, n, get_even_scaling_factor_cubic_bspline_wavelets(), level, 1, stride, cyclical);
    update(sample, n, get_first_update_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    predict(sample, n, get_prediction_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_cubic_bspline_wavelets(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_second_update_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    iupdate(sample, n, get_first_update_mask_cubic_bspline_wavelets(), level, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_cubic_bspline_wavelets(), level, 1, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_cubic_bsplines()
    {
    static std::vector<double> mask = compute_symmetric_mask(0.5);
    return mask;
    }

  inline const std::vector<double>& get_update_mask_cubic_bsplines()
    {
    static std::vector<double> mask = compute_symmetric_mask(-0.25);
    return mask;
    }

  inline double get_even_scaling_factor_cubic_bsplines()
    {
    return 2.0;
    }

  template <typename T>
  void forward_cubic_bsplines(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    update(sample, n, get_update_mask_cubic_bsplines(), level, stride, cyclical);
    scale_even(sample, n, get_even_scaling_factor_cubic_bsplines(), level, 1, stride, cyclical);
    predict(sample, n, get_prediction_mask_cubic_bsplines(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_cubic_bsplines(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    ipredict(sample, n, get_prediction_mask_cubic_bsplines(), level, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_cubic_bsplines(), level, 1, stride, cyclical);
    iupdate(sample, n, get_update_mask_cubic_bsplines(), level, stride, cyclical);
    }

  inline std::vector<double> compute_prediction_mask_4_point()
    {
    std::vector<double> prediction_mask;
    prediction_mask.reserve(4);
    prediction_mask.push_back(-1.0 / 16.0);
    prediction_mask.push_back(9.0 / 16.0);
    prediction_mask.push_back(9.0 / 16.0);
    prediction_mask.push_back(-1.0 / 16.0);
    return prediction_mask;
    }

  inline const std::vector<double>& get_prediction_mask_4_point()
    {
    static std::vector<double> prediction_mask(compute_prediction_mask_4_point());
    return prediction_mask;
    }

  inline const std::vector<double> & get_update_mask_4_point()
    {
    //return 1.0 / 4.0;
    static std::vector<double> mask = compute_symmetric_mask(0.25);
    return mask;
    }

  template <typename T>
  void forward_4_point(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_prediction_mask_4_point(), level, stride, cyclical);
    update(sample, n, get_update_mask_4_point(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_4_point(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_update_mask_4_point(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_4_point(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_cdf_5_3()
    {
    //return 0.5;
    static std::vector<double> mask = compute_symmetric_mask(0.5);
    return mask;
    }

  inline const std::vector<double>& get_update_mask_cdf_5_3()
    {
    //return 0.25;
    static std::vector<double> mask = compute_symmetric_mask(0.25);
    return mask;
    }

  template <typename T>
  void forward_cdf_5_3(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_prediction_mask_cdf_5_3(), level, stride, cyclical);
    update(sample, n, get_update_mask_cdf_5_3(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_cdf_5_3(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_update_mask_cdf_5_3(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_cdf_5_3(), level, stride, cyclical);
    }

  inline std::vector<double> compute_prediction_mask_daubechies_d4()
    {
    std::vector<double> prediction_mask;
    prediction_mask.reserve(4);
    prediction_mask.push_back(((sqrt(3.0) - 2.0) / 4.0));
    prediction_mask.push_back((sqrt(3.0) / 4.0));
    prediction_mask.push_back(0.0);
    prediction_mask.push_back(0.0);
    return prediction_mask;
    }

  inline const std::vector<double>& get_prediction_mask_daubechies_d4()
    {
    static std::vector<double> prediction_mask(compute_prediction_mask_daubechies_d4());
    return prediction_mask;
    }

  inline std::vector<double> compute_first_update_mask_daubechies_d4()
    {
    std::vector<double> update_mask;
    update_mask.reserve(2);
    update_mask.push_back(0.0);
    update_mask.push_back(sqrt(3.0));
    return update_mask;
    }

  inline const std::vector<double>& get_first_update_mask_daubechies_d4()
    {
    static std::vector<double> update_mask(compute_first_update_mask_daubechies_d4());
    return update_mask;
    }

  inline std::vector<double> compute_second_update_mask_daubechies_d4()
    {
    std::vector<double> update_mask;
    update_mask.reserve(4);
    update_mask.push_back(0.0);
    update_mask.push_back(0.0);
    update_mask.push_back(0.0);
    update_mask.push_back((-1.0));
    return update_mask;
    }

  inline const std::vector<double>& get_second_update_mask_daubechies_d4()
    {
    static std::vector<double> update_mask(compute_second_update_mask_daubechies_d4());
    return update_mask;
    }

  inline double get_even_scaling_factor_daubechies_d4()
    {
    return ((sqrt(3.0) - 1.0) / (2.0));
    }

  inline double get_odd_scaling_factor_daubechies_d4()
    {
    return ((sqrt(3.0) + 1.0) / (2.0));
    }

  template <typename T>
  void forward_daubechies_d4(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    update(sample, n, get_first_update_mask_daubechies_d4(), level, stride, cyclical);
    predict(sample, n, get_prediction_mask_daubechies_d4(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_daubechies_d4(), level, stride, cyclical);
    scale_even(sample, n, get_even_scaling_factor_daubechies_d4(), level, 0, stride, cyclical);
    scale_odd(sample, n, get_odd_scaling_factor_daubechies_d4(), level, 0, stride, cyclical);
    }

  template <typename T>
  void inverse_daubechies_d4(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iscale_odd(sample, n, get_odd_scaling_factor_daubechies_d4(), level, 0, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_daubechies_d4(), level, 0, stride, cyclical);
    iupdate(sample, n, get_second_update_mask_daubechies_d4(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_daubechies_d4(), level, stride, cyclical);
    iupdate(sample, n, get_first_update_mask_daubechies_d4(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_haar()
    {
    static std::vector<double> prediction_mask((size_t)1, 1.0);
    return prediction_mask;
    }

  inline std::vector<double> compute_update_mask_haar()
    {
    std::vector<double> update_mask;
    update_mask.reserve(2);
    update_mask.push_back(0.5);
    update_mask.push_back(0.0);
    return update_mask;
    }

  inline const std::vector<double>& get_update_mask_haar()
    {
    static std::vector<double> update_mask(compute_update_mask_haar());
    return update_mask;
    }

  template <typename T>
  void forward_haar(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_prediction_mask_haar(), level, stride, cyclical);
    update(sample, n, get_update_mask_haar(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_haar(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_update_mask_haar(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_haar(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_first_prediction_mask_cdf_9_7()
    {
    //return 1.5861343420693648;
    static std::vector<double> mask = compute_symmetric_mask(1.5861343420693648);
    return mask;
    }

  inline const std::vector<double>& get_first_update_mask_cdf_9_7()
    {
    //return -0.0529801185718856;
    static std::vector<double> mask = compute_symmetric_mask(-0.0529801185718856);
    return mask;
    }

  inline const std::vector<double>& get_second_prediction_mask_cdf_9_7()
    {
    //return -0.8829110755411875;
    static std::vector<double> mask = compute_symmetric_mask(-0.8829110755411875);
    return mask;
    }

  inline const std::vector<double>& get_second_update_mask_cdf_9_7()
    {
    //return 0.4435068520511142;
    static std::vector<double> mask = compute_symmetric_mask(0.4435068520511142);
    return mask;
    }

  inline double get_odd_scaling_factor_cdf_9_7()
    {
    return 1.0 / 1.6257861322319229;
    }

  inline double get_even_scaling_factor_cdf_9_7()
    {
    return 1.0 / 1.230174104914126;
    }

  template <typename T>
  void forward_cdf_9_7(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_first_prediction_mask_cdf_9_7(), level, stride, cyclical);
    update(sample, n, get_first_update_mask_cdf_9_7(), level, stride, cyclical);
    predict(sample, n, get_second_prediction_mask_cdf_9_7(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_cdf_9_7(), level, stride, cyclical);
    scale_odd(sample, n, get_odd_scaling_factor_cdf_9_7(), level, 0, stride, cyclical);
    scale_even(sample, n, get_even_scaling_factor_cdf_9_7(), level, 0, stride, cyclical);
    }

  template <typename T>
  void inverse_cdf_9_7(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iscale_even(sample, n, get_even_scaling_factor_cdf_9_7(), level, 0, stride, cyclical);
    iscale_odd(sample, n, get_odd_scaling_factor_cdf_9_7(), level, 0, stride, cyclical);
    iupdate(sample, n, get_second_update_mask_cdf_9_7(), level, stride, cyclical);
    ipredict(sample, n, get_second_prediction_mask_cdf_9_7(), level, stride, cyclical);
    iupdate(sample, n, get_first_update_mask_cdf_9_7(), level, stride, cyclical);
    ipredict(sample, n, get_first_prediction_mask_cdf_9_7(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_jamlet_linear()
    {
    //return 0.5;
    static std::vector<double> mask = compute_symmetric_mask(0.5);
    return mask;
    }

  inline std::vector<double> compute_update_mask_jamlet_linear()
    {
    std::vector<double> update_mask;
    update_mask.reserve(4);
    update_mask.push_back(-0.0562);
    update_mask.push_back(0.3062);
    update_mask.push_back(0.3062);
    update_mask.push_back(-0.0562);
    return update_mask;
    }

  inline const std::vector<double>& get_update_mask_jamlet_linear()
    {
    static std::vector<double> update_mask(compute_update_mask_jamlet_linear());
    return update_mask;
    }

  template <typename T>
  void forward_jamlet_linear(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_prediction_mask_jamlet_linear(), level, stride, cyclical);
    update(sample, n, get_update_mask_jamlet_linear(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_jamlet_linear(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_update_mask_jamlet_linear(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_jamlet_linear(), level, stride, cyclical);
    }

  inline std::vector<double> compute_prediction_mask_jamlet_quadratic()
    {
    std::vector<double> prediction_mask;
    prediction_mask.reserve(2);
    prediction_mask.push_back(0.25);
    prediction_mask.push_back(0.75);
    return prediction_mask;
    }

  inline const std::vector<double>& get_prediction_mask_jamlet_quadratic()
    {
    static std::vector<double> prediction_mask(compute_prediction_mask_jamlet_quadratic());
    return prediction_mask;
    }

  inline std::vector<double> compute_first_update_mask_jamlet_quadratic()
    {
    std::vector<double> update_mask;
    update_mask.reserve(2);
    update_mask.push_back(0.0);
    update_mask.push_back(-1.0 / 3.0);
    return update_mask;
    }

  inline const std::vector<double>& get_first_update_mask_jamlet_quadratic()
    {
    static std::vector<double> update_mask(compute_first_update_mask_jamlet_quadratic());
    return update_mask;
    }

  inline std::vector<double> compute_second_update_mask_jamlet_quadratic()
    {
    std::vector<double> update_mask;
    update_mask.reserve(4);
    update_mask.push_back(-0.0975);
    update_mask.push_back(0.430833333333);
    update_mask.push_back(0.430833333333);
    update_mask.push_back(-0.0975);
    return update_mask;
    }

  inline const std::vector<double>& get_second_update_mask_jamlet_quadratic()
    {
    static std::vector<double> update_mask(compute_second_update_mask_jamlet_quadratic());
    return update_mask;
    }

  inline double get_even_scaling_factor_jamlet_quadratic()
    {
    return 3.0 / 2.0;
    }

  template <typename T>
  void forward_jamlet_quadratic(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    update(sample, n, get_first_update_mask_jamlet_quadratic(), level, stride, cyclical);
    scale_even(sample, n, get_even_scaling_factor_jamlet_quadratic(), level, 1, stride, cyclical);
    predict(sample, n, get_prediction_mask_jamlet_quadratic(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_jamlet_quadratic(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_jamlet_quadratic(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_second_update_mask_jamlet_quadratic(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_jamlet_quadratic(), level, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_jamlet_quadratic(), level, 1, stride, cyclical);
    iupdate(sample, n, get_first_update_mask_jamlet_quadratic(), level, stride, cyclical);
    }

  inline const std::vector<double>& get_prediction_mask_jamlet_cubic()
    {
    //return 0.5;
    static std::vector<double> mask = compute_symmetric_mask(0.5);
    return mask;
    }

  inline const std::vector<double>& get_first_update_mask_jamlet_cubic()
    {
    //return -0.5;
    static std::vector<double> mask = compute_symmetric_mask(-0.5);
    return mask;
    }

  inline std::vector<double> compute_second_update_mask_jamlet_cubic()
    {
    std::vector<double> update_mask;
    update_mask.reserve(4);
    update_mask.push_back(-0.1217);
    update_mask.push_back(0.4967);
    update_mask.push_back(0.4967);
    update_mask.push_back(-0.1217);
    return update_mask;
    }

  inline const std::vector<double>& get_second_update_mask_jamlet_cubic()
    {
    static std::vector<double> update_mask(compute_second_update_mask_jamlet_cubic());
    return update_mask;
    }

  inline double get_even_scaling_factor_jamlet_cubic()
    {
    return 2.0;
    }

  template <typename T>
  void forward_jamlet_cubic(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    scale_even(sample, n, get_even_scaling_factor_jamlet_cubic(), level, 1, stride, cyclical);
    update(sample, n, get_first_update_mask_jamlet_cubic(), level, stride, cyclical);
    predict(sample, n, get_prediction_mask_jamlet_cubic(), level, stride, cyclical);
    update(sample, n, get_second_update_mask_jamlet_cubic(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_jamlet_cubic(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_second_update_mask_jamlet_cubic(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_jamlet_cubic(), level, stride, cyclical);
    iupdate(sample, n, get_first_update_mask_jamlet_cubic(), level, stride, cyclical);
    iscale_even(sample, n, get_even_scaling_factor_jamlet_cubic(), level, 1, stride, cyclical);
    }

  inline std::vector<double> compute_prediction_mask_jamlet_4_point()
    {
    std::vector<double> prediction_mask;
    prediction_mask.reserve(4);
    prediction_mask.push_back(-1.0 / 16.0);
    prediction_mask.push_back(9.0 / 16.0);
    prediction_mask.push_back(9.0 / 16.0);
    prediction_mask.push_back(-1.0 / 16.0);
    return prediction_mask;
    }

  inline const std::vector<double>& get_prediction_mask_jamlet_4_point()
    {
    static std::vector<double> prediction_mask(compute_prediction_mask_jamlet_4_point());
    return prediction_mask;
    }

  inline std::vector<double> compute_update_mask_jamlet_4_point()
    {
    std::vector<double> update_mask;
    update_mask.reserve(4);
    update_mask.push_back(-0.0415);
    update_mask.push_back(0.2915);
    update_mask.push_back(0.2915);
    update_mask.push_back(-0.0415);
    return update_mask;
    }

  inline const std::vector<double>& get_update_mask_jamlet_4_point()
    {
    static std::vector<double> update_mask(compute_update_mask_jamlet_4_point());
    return update_mask;
    }

  template <typename T>
  void forward_jamlet_4_point(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    predict(sample, n, get_prediction_mask_jamlet_4_point(), level, stride, cyclical);
    update(sample, n, get_update_mask_jamlet_4_point(), level, stride, cyclical);
    }

  template <typename T>
  void inverse_jamlet_4_point(T* sample, uint64_t n, uint64_t level, uint64_t stride = 1, bool cyclical = false)
    {
    iupdate(sample, n, get_update_mask_jamlet_4_point(), level, stride, cyclical);
    ipredict(sample, n, get_prediction_mask_jamlet_4_point(), level, stride, cyclical);
    }
  }
