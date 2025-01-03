#ifndef BTACH_VECTOR_H
#define BTACH_VECTOR_H
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <math.h>


class BatchVector
{
  private:
    xt::xarray<double> cache_data;
    bool initialized = false;
    int length = 0;
  public:
    int clock_step = 0;
    BatchVector();
    BatchVector(int length);
    ~BatchVector(void);
    bool is_initialized();
    xt::xarray<double> data();
    xt::xarray<double> row_element(int index);
    xt::xarray<double> col_element(int index);
    void update(double x,double y,double z);
    void update(xt::xarray<double> new_vector);
};
#endif 