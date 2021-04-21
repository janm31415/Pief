#pragma once

#include "tokenize.h"

#include <variant>

typedef std::vector<double> values;

class Factor;
class Float;

template<typename T>
class Precedence { public: std::vector<T> operands; std::vector<std::string> fops; int line_nr; };


typedef Precedence<Factor> Term;
typedef Precedence<Term> Expression;

class Tag { public: std::string name; int line_nr; };

typedef std::variant<Expression, Tag> Statement;
typedef std::vector<Statement> Statements;

class FuncCall { public: std::string name; std::vector<Expression> exprs; int line_nr; };
class Factor { public: char sign = '+'; std::variant<double, Expression, FuncCall> factor; int line_nr; };

class Program { public: Statements statements; };

Program make_program(tokens& tokes);

double get_value(const Expression& e);