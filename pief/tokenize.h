#pragma once

#include <string>
#include <stdint.h>
#include <vector>

double to_double(const char* value);
bool is_number(bool& is_real, const char* value);

struct token
  {
  enum e_type
    {
    T_BAD,
    T_LEFT_ROUND_BRACKET,
    T_RIGHT_ROUND_BRACKET,    
    T_INTEGER,
    T_REAL,
    T_ID,   
    T_PLUS,
    T_MINUS,
    T_MUL,
    T_DIV,
    T_SEMICOLON
    };

  e_type type;
  std::string value;
  int line_nr;

  token(e_type i_type, const std::string& v, int i_line_nr) : type(i_type), value(v), line_nr(i_line_nr) {}
  };

typedef std::vector<token> tokens;
tokens tokenize(const std::string& str);
