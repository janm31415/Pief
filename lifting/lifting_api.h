#pragma once

#if defined(WIN32) || defined(_WIN32)
#  if defined(lifting_EXPORTS)
#    define LIFTING_API __declspec(dllexport)
#  else
#    define LIFTING_API __declspec(dllimport)
#  endif
#else
#  define LIFTING_API
#endif
