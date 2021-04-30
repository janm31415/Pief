#include "parse.h"

#include <sstream>
#include <functional>
#include <map>
#include <cmath>

namespace
  {

  void throw_parse_error(int line_nr, const std::string& message)
    {
    if (line_nr <= 0)
      throw std::logic_error("parse error: " + message);
    std::stringstream str;
    str << line_nr;
    throw std::logic_error("parse error: line " + str.str() + ": " + message);
    }

  std::string current(const tokens& tokens)
    {
    return tokens.empty() ? std::string() : tokens.back().value;
    }

  token take(tokens& tokens)
    {
    if (tokens.empty())
      {
      throw_parse_error(-1, "unexpected end");
      }
    token t = tokens.back();
    tokens.pop_back();
    return t;
    }

  token::e_type current_type(const tokens& tokens)
    {
    return tokens.empty() ? token::T_BAD : tokens.back().type;
    }

  void advance(tokens& tokens)
    {
    tokens.pop_back();
    }

  void require(tokens& tokens, std::string required)
    {
    if (tokens.empty())
      {
      throw_parse_error(-1, "unexpected end: missing " + required);
      }
    auto t = take(tokens);
    if (t.value != required)
      {
      throw_parse_error(t.line_nr, "required: " + required + ", found: " + t.value);
      }
    }

  bool is_tag(const std::string& fun)
    {
    return fun == "predict" || fun == "update" || fun == "scale_even" || fun == "scale_odd";
    }

  bool is_fun(const std::string& fun)
    {
    return fun == "sqrt" || fun == "pow" || fun == "sin" || fun == "log" || fun == "exp" || fun == "log2" || fun == "abs";
    }

  Expression make_expression(tokens& tokes);

  Tag make_tag(tokens& tokes)
    {
    Tag tag;
    auto t = take(tokes);
    tag.line_nr = t.line_nr;
    tag.name = t.value;
    return tag;
    }

  FuncCall make_funccall(token t, tokens& tokes)
    {
    FuncCall fn;
    fn.line_nr = t.line_nr;
    fn.name = t.value;
    if (!is_fun(fn.name))
      throw_parse_error(t.line_nr, "Unknown function " + fn.name);
    require(tokes, "(");
    if (current(tokes) != ")")
      {
      fn.exprs.push_back(make_expression(tokes));
      while (current(tokes) == ",")
        {
        advance(tokes);
        fn.exprs.push_back(make_expression(tokes));
        }
      }
    require(tokes, ")");
    return fn;
    }

  Factor make_factor(tokens& tokes)
    {
    Factor f;
    token toke = take(tokes);
    f.line_nr = toke.line_nr;

    // optional sign
    if ((toke.value == "+") || (toke.value == "-"))
      {
      f.sign = toke.value[0];
      toke = take(tokes);
      }

    switch (toke.type) {
      case token::T_LEFT_ROUND_BRACKET:
        f.factor = make_expression(tokes);
        require(tokes, ")");
        break;
      case token::T_REAL:
        f.factor = to_double(toke.value.c_str());
        break;
      case token::T_INTEGER:
        f.factor = to_double(toke.value.c_str());
        break;
      case token::T_ID:
        f.factor = make_funccall(toke, tokes);
        break;
      default:
        throw_parse_error(toke.line_nr, "Unhandled type");
      }
    return f;
    }

  template <typename T, typename U>
  T parse_multiop(tokens& tokes, std::function<U(tokens&)> make, std::vector<std::string> ops)
    {
    T node;
    node.line_nr = tokes.back().line_nr;
    node.operands.push_back(make(tokes));
    while (1) {
      std::string op = current(tokes);
      std::vector<std::string>::iterator opit = std::find(ops.begin(), ops.end(), op);
      if (opit == ops.end())
        break;
      tokes.pop_back();
      node.fops.push_back(op);
      node.operands.push_back(make(tokes));
      }
    return node;
    }

  Term make_term(tokens& tokes) { return parse_multiop<Term, Factor>(tokes, make_factor, { "*", "/" }); }

  Expression make_expression(tokens& tokes)
    {
    return parse_multiop<Expression, Term>(tokes, make_term, { "+", "-" });
    }

  Statement make_statement(tokens& tokes)
    {
    if (tokes.empty())
      throw_parse_error(-1, "incomplete tokens stack");
    Statement stm;
    if (tokes.back().type == token::T_ID)
      {
      if (is_tag(current(tokes)))
        return make_tag(tokes);
      }
    return make_expression(tokes);
    }
  }

Program make_program(tokens& tokes)
  {
  std::reverse(tokes.begin(), tokes.end());
  Program prog;
  while (!tokes.empty())
    {
    prog.statements.push_back(make_statement(tokes));
    require(tokes, ";");
    }
  return prog;
  }

namespace
  {

  typedef std::vector<double> values;
  typedef std::function<double(values)> c_func;
  typedef std::map<std::string, c_func> c_funcs_t;

  double c_add(values v)
    {
    return v[0] + v[1];
    }

  double c_sub(values v)
    {
    return v[0] - v[1];
    }

  double c_mul(values v)
    {
    return v[0] * v[1];
    }

  double c_div(values v)
    {
    return v[0] / v[1];
    }

  double c_sqrt(values v)
    {
    return std::sqrt(v[0]);
    }

  double c_pow(values v)
    {
    return pow(v[0], v[1]);
    }

  double c_sin(values v)
    {
    return std::sin(v[0]);
    }

  double c_cos(values v)
    {
    return std::cos(v[0]);
    }


  double c_exp(values v)
    {
    return std::exp(v[0]);
    }

  double c_log(values v)
    {
    return std::log(v[0]);
    }

  double c_log2(values v)
    {
    return std::log2(v[0]);
    }

  double c_abs(values v)
    {
    return std::abs(v[0]);
    }

  c_funcs_t c_funcs =
    {
    {"+", c_add},
    {"-", c_sub},
    {"*", c_mul},
    {"/", c_div},
    {"sqrt", c_sqrt},
    {"pow", c_pow},
    {"sin", c_sin},
    {"cos", c_cos},
    {"exp", c_exp},
    {"log", c_log},
    {"log2", c_log2},
    {"abs", c_abs}
    };

  template <class T>
  double _compute(const T& op)
    {
    double value = 0.0;
    int j = 0;
    while (j < op.fops.size())
      {
      if (is_constant(op.operands[j]) && is_constant(op.operands[j + 1]))
        {
        auto v1 = get_constant_value(op.operands[j]);
        auto v2 = get_constant_value(op.operands[j + 1]);
        auto it = c_funcs.find(op.fops[j]);
        if (it == c_funcs.end())
          throw std::runtime_error("error: unknown function: " + op.fops[j]);
        c_func f = it->second;
        values v;
        v.reserve(2);
        v.push_back(v1);
        v.push_back(v2);
        auto res = f(v);
        op.fops.erase(op.fops.begin() + j);
        op.operands.erase(op.operands.begin() + j);
        set_constant_value(op.operands[j], res);
        --j;
        }
      ++j;
      }
    }
  }

double get_value(const Expression& e);

double get_value(const FuncCall& f)
  {
  values v;
  for (const auto& e : f.exprs)
    v.push_back(get_value(e));
  auto it = c_funcs.find(f.name);
  if (it == c_funcs.end())
    throw std::logic_error("error: unknown function: " + f.name);
  c_func fun = it->second;
  return fun(v);
  }

double get_value(const Factor& f)
  {
  double ret;
  if (std::holds_alternative<double>(f.factor))
    ret = std::get<double>(f.factor);
  else if (std::holds_alternative<FuncCall>(f.factor))
    ret = get_value(std::get<FuncCall>(f.factor));
  else
    ret = get_value(std::get<Expression>(f.factor));
  if (f.sign == '-')
    {
    ret = -ret;
    }
  return ret;
  }

double get_value(const Term& t)
  {
  int j = 0;
  double value = get_value(t.operands[j]);
  while (j < t.fops.size())
    {
    auto v2 = get_value(t.operands[j + 1]);
    auto it = c_funcs.find(t.fops[j]);
    if (it == c_funcs.end())
      throw std::logic_error("error: unknown function: " + t.fops[j]);
    c_func f = it->second;
    values v;
    v.reserve(2);
    v.push_back(value);
    v.push_back(v2);
    value = f(v);
    ++j;
    }
  return value;
  }

double get_value(const Expression& e)
  {
  int j = 0;
  double value = get_value(e.operands[j]);
  while (j < e.fops.size())
    {
    auto v2 = get_value(e.operands[j + 1]);
    auto it = c_funcs.find(e.fops[j]);
    if (it == c_funcs.end())
      throw std::logic_error("error: unknown function: " + e.fops[j]);
    c_func f = it->second;
    values v;
    v.reserve(2);
    v.push_back(value);
    v.push_back(v2);
    value = f(v);
    ++j;
    }
  return value;
  }
