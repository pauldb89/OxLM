#pragma once

// BOOST_CLASS_EXPORT does the magic required to correctly serialize/deserialize
// derived classes through pointers of the base class. However, it needs to know
// in advance what archives are used to serialize the derived classes.
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/export.hpp>
