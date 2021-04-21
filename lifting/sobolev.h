#pragma once

#include "lifting_api.h"

#include "jtk/jtk/mat.h"
#include <vector>


LIFTING_API jtk::mat transop(const std::vector<double>& P);
LIFTING_API void sumruleorder(int& SMO, double& P0, const std::vector<double>& P);
LIFTING_API double sobsmthest(const std::vector<double>& P);