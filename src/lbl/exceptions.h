#pragma once

#include <exception>

using namespace std;

namespace oxlm {

class UnknownModelException : public exception {
  virtual const char* what() const throw() {
    return "Unknown model type";
  }
};

class UnknownActivationException : public exception {
  virtual const char* what() const throw() {
    return "Unknown activation type";
  }
};

} // namespace oxlm
