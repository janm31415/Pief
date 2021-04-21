#include "sobolev.h"
#include <numeric>

jtk::mat transop(const std::vector<double>& p)
  {
  const int m = 2;
  const int N = (int)p.size() - 1;
  const int L = N;

  std::vector a((size_t)(2 * N + 1), 0.0);
  
  for (int j1 = 0; j1 < 2 * N + 1; ++j1)
    {
    for (int ell1 = std::max<int>(0, j1 - N); ell1 <= std::min<int>(N, j1); ++ell1)
      a[j1] += p[N + ell1 - j1] * p[ell1];
    }

  jtk::mat T(2*L+1, 2*L+1);
  for (int i1 = 1; i1 <= 2 * L + 1; ++i1)
    {
    for (int j1 = std::max<int>(1, m*(i1 - L - 1) + L + 1 - N); j1 <=
      std::min<int>(m*(i1 - L - 1) + N + L + 1, 2 * L + 1); ++j1)
      {
      T(i1 - 1, j1 - 1) = a[m*(i1 - L - 1) - j1 + L + 2 + N - 1]/m;
      }
    }  
  return T;
  }

namespace
  {
  double factorial(int f)
    {
    int fact = 1;
    for (int i = 2; i <= f; ++i)
      fact = fact * i;
    return (double)fact;
    }
  }

void sumruleorder(int& SLO, double& P0, const std::vector<double>& p)
  {
  std::vector<double> ELp = { {1.0, -1.0} };
  const int N = (int)p.size() - 1;
  const int m = 2;
  std::vector<double> P = p;
  for (auto& v : P)
    v /= (double)m;
  SLO = 0;
  std::vector<double> PP((size_t)m, 0.0);

  for (int j1 = 0; j1 <= N; ++j1)
    {
    for (int k = 0; k < m; ++k)
      {
      PP[k] = PP[k] + pow(ELp[k], j1)*P[j1];
      }
    }
  P0 = PP[0];
  if (std::abs(P0 - 1.0) > 1e-4)
    throw std::logic_error("P0 does not satisfy condition E");
  //check if the mask satisfies the sum rule of order at least 1
  double vpp = PP[1];
  const double error = 1e-8;
  if (std::abs(vpp) > error)
    throw std::logic_error("Mask does not satisfy the sum rule of order 1");
  SLO = 1;
  int slo0 = 10;
  int k0 = slo0;
  jtk::mat DP = jtk::zeros(k0 + 1, m);
  DP(0, 1) = PP[1];
  jtk::mat y = jtk::zeros(k0 + 1, 1);
  y(0) = 1.0;


  for (int i2 = 0; i2 <= k0; ++i2)
    {
    for (int k = 1; k <= m; ++k)
      {
      for (int j2 = 0; j2 <= N; ++j2)
        {
        DP(i2, k - 1) += pow((double)j2, (double)i2) * pow(ELp[k - 1], (double)j2) * P[j2];
        }
      }
    }
  //check sumrule order 2
  y(1) = (-1.0)*y(0)*DP(1, 0) / (1.0 - m * P0);

  double sum00 = m * y(1)*PP[1] + (-1)*y(0)*DP(1, 1);
  if (std::abs(sum00) > error)
    return;
  SLO = 2;

  //check sum rule order greater than 2 
  jtk::mat lesssum00r = jtk::zeros(m, 1);
  for (int s = 2; s <= k0; ++s)
    {
    jtk::mat sumpir = jtk::zeros(m, 1);
    for (int k = 1; k <= m; ++k)
      {
      for (int ga1 = 0; ga1 < s; ++ga1)
        {        
        lesssum00r(k - 1) += factorial(s) / (factorial(ga1)*factorial(s - ga1))*pow(m, ga1)*pow(-1, s - ga1)*y(ga1)*DP(s - ga1, k - 1);
        }
      y(s) = lesssum00r(0) / (1.0 - pow(m, s) * P0);
      sumpir(k - 1) = lesssum00r(k - 1) + pow(m, s)*y(s)*PP[k - 1];
      }
    double sumpirmax = std::abs(sumpir(1));
    if (sumpirmax > error)
      return;
    ++SLO;
    }
  }

namespace
  {  
  template <class TIter, class TFunc>
  void sort(std::vector<double>& sorted, std::vector<size_t>& indices, TIter first, TIter last, TFunc comp)
    {
    size_t sz = std::distance(first, last);
    indices.resize(sz);
    std::iota(indices.begin(), indices.end(), (size_t)0);
    std::sort(indices.begin(), indices.end(), [&](size_t left, size_t right)
      {
      return comp(*(first + left), *(first + right));
      });
    sorted.resize(sz);
    for (size_t i = 0; i < sz; ++i)
      sorted[i] = *(first + indices[i]);    
    }

  template <class TIter>
  void sort(std::vector<double>& sorted, std::vector<size_t>& indices, TIter first, TIter last)
    {
    sort(sorted, indices, first, last, [](typename TIter::value_type left, typename TIter::value_type right) { return left < right; });
    }
  }

double sobsmthest(const std::vector<double>& P)
  {
  double reg = 0.0;
  int SLO;
  double P_0;
  sumruleorder(SLO, P_0, P);
  auto T = transop(P);

  int vsize = 2 * SLO;

  jtk::mat wr, wi;
  jtk::eig(T, wr, wi);

  for (uint32_t j = 0; j < wr.rows(); ++j)
    {
    if (std::abs(wi(j)) > 1e-6)
      wr(j) = 0.0;
    }
  
  std::vector<double> sreigvalue;
  std::vector<size_t> od;
  sort(sreigvalue, od, wr.begin(), wr.end(), [](double left, double right) {return std::abs(left) > std::abs(right); });

  std::vector<double> bigreig((size_t)vsize);
  for (size_t i = 0; i < (size_t)vsize; ++i)
    bigreig[i] = wr((uint32_t)od[i + 1]);

  std::vector<double> eigen1((size_t)vsize, 0.0);
  for (int j = 0; j < 2 * SLO - 1; ++j)
    eigen1[j] = 1.0 / pow(2.0, j+1.0);

  /*
  std::vector<double> sbigreig;
  std::vector<size_t > od1;
  sort(sbigreig, od1, bigreig.begin(), bigreig.end());
  */
  std::sort(bigreig.begin(), bigreig.end());
  std::reverse(eigen1.begin(), eigen1.end());

  std::vector<double> difference(eigen1.size());
  for (size_t i = 0; i < difference.size(); ++i)
    difference[i] = bigreig[i] - eigen1[i];

  const double err2 = 1e-6;

  int j0 = 0;
  int j_0 = 0;
  for (int j = 0; j < vsize; ++j)
    {
    if (std::abs(difference[j]) > std::min<double>(err2, 1.0/pow(2.0, j+3.0)))
      {
      j0 = j;
      break;
      }
    }
  for (int j = vsize-1; j >= 0; --j)
    {
    if (std::abs(difference[j]) > std::min<double>(err2, 1.0 / pow(2.0, j + 3.0)))
      {
      j_0 = j;
      break;
      }
    }
  reg = std::max<double>(std::abs(bigreig[j0]), std::abs(bigreig[j_0]));
  reg = -log(reg) / log(2.0) / 2.0;
  return reg;
  }