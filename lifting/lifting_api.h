#pragma once

#if defined(lifting_EXPORTS)
#  define LIFTING_API __declspec(dllexport)
#else
#  define LIFTING_API __declspec(dllimport)
#endif