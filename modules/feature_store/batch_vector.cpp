#include  "batch_vector.h"
#include <vector>

BatchVector::BatchVector()
{
    this->cache_data = xt::empty<double>({0});
    this->length = 0;
}

BatchVector::BatchVector(int length)
{
    std::vector<int>  shape = {length,3};
    this->cache_data = xt::zeros<double>(shape);
    this->length = length;
}

BatchVector::~BatchVector(void)
{
    this->initialized = false;
    this-> clock_step = 0;
}

bool BatchVector::is_initialized()
{
    return this->initialized;
}

xt::xarray<double> BatchVector::data()
{
    return  this->cache_data;
}
xt::xarray<double> BatchVector::row_element(int index)
{
    return  xt::row(this->cache_data,index);
}
    
xt::xarray<double> BatchVector::col_element(int index)
{
    return xt::col(this->cache_data,index);
}

void BatchVector::update(double x,double y,double z)
{
    // first element is the latest element 
    this->clock_step = this->clock_step +1;
    if (this->clock_step > this->length)
    {
        this->initialized = true;
    }
    auto old_data_part = xt::view(this->cache_data,xt::range(0,this->length-1),xt::all());
    auto new_part = xt::xarray<double>({x,y,z});
    xt::view(this->cache_data,xt::range(1,this->length),xt::all()) = old_data_part;
    xt::view(this->cache_data,0,xt::all()) = new_part;
}
void BatchVector::update(xt::xarray<double> new_vector)
{
    // first element is the latest element 
    this->clock_step = this->clock_step +1;
    if (this->clock_step > this->length)
    {
        this->initialized = true;
    }
    auto old_data_part = xt::view(this->cache_data,xt::range(0,this->length-1),xt::all());
    xt::view(this->cache_data,xt::range(1,this->length),xt::all()) = old_data_part;
    xt::view(this->cache_data,0,xt::all()) = new_vector;
}