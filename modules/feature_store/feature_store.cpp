#include "feature_store.h"

Feature_Store::Feature_Store(
    double deltaT,
    int based_window,
    int cache_length
) : deltaT(deltaT),
    based_window(based_window),
    cache_length(cache_length),
    track_initialized(false),
    image_initialized(false)
{
    this->Observe = new BatchVector(cache_length);
    this->Filter_P = new BatchVector(cache_length);
    this->Filter_V = new BatchVector(cache_length);
    this->Filter_a = new BatchVector(cache_length);
    this->Base_Vector = new BatchVector(cache_length);
    this->Filter_V_Target = new BatchVector(cache_length);
    this->Filter_a_Target = new BatchVector(cache_length);
    this->Filter_x_Target = new BatchVector(cache_length);
}

Feature_Store::~Feature_Store()
{
  delete  Observe;
  delete  Filter_P;
  delete  Filter_V;
  delete  Filter_a;
  delete  Base_Vector;
  delete  Filter_x_Target; 
  delete  Filter_V_Target;
  delete  Filter_a_Target;
}

void Feature_Store::update(          
    double Observe_x,
    double Observe_y,
    double Observe_z,
    double Filter_P_x,
    double Filter_P_y,
    double Filter_P_z,
    double Filter_V_x,
    double Filter_V_y,
    double Filter_V_z,
    double Filter_a_x,
    double Filter_a_y,
    double Filter_a_z
)
{
    
    this->Observe->update(Observe_x,Observe_y,Observe_z);
    this->Filter_P->update(Filter_P_x,Filter_P_y,Filter_P_z);
    this->Filter_V->update(Filter_V_x,Filter_V_y,Filter_V_z);
    this->Filter_a->update(Filter_a_x,Filter_a_y,Filter_a_z);
    if(this->Filter_P->clock_step >= this->based_window)
    {
        xt::xarray<double>  current_base = this->Sub(
          this->Mul(this->Filter_P->row_element(0),1.0/this->deltaT/this->based_window),
          this->Mul(this->Filter_P->row_element(this->based_window-1),1.0/this->deltaT/this->based_window)
        );
        this->Base_Vector->update(current_base);
        this->Filter_x_Target->update(
          this->Real2Target(
            this->Filter_P->row_element(0),
            this->Base_Vector->row_element(0)
          )
        );
        //compute 
        this->Filter_V_Target->update(
          this->Real2Target(
            this->Filter_V->row_element(0),
            this->Base_Vector->row_element(0)
          )
        );
        this->Filter_a_Target->update(
          this->Real2Target(
            this->Filter_a->row_element(0),
            this->Base_Vector->row_element(0)
          )
        );
    }
}

bool Feature_Store::is_track_initialized() const {
    return this->Observe->is_initialized() && 
           this->Filter_P->is_initialized() && 
           this->Filter_V->is_initialized() && 
           this->Filter_a->is_initialized() &&
           this->Base_Vector->is_initialized() &&
           this->Filter_x_Target->is_initialized() &&
           this->Filter_V_Target->is_initialized() &&
           this->Filter_a_Target->is_initialized();
}

bool Feature_Store::is_image_initialized() const {
    return image_initialized;
}

bool Feature_Store::is_fully_initialized() const {
    return is_track_initialized() && is_image_initialized();
}

void Feature_Store::compute_smooth_features(int smooth_window,
    const xt::xarray<double>& filter_v_target,
    const xt::xarray<double>& filter_a_target,
    xt::xarray<double>& smooth_stdv,
    xt::xarray<double>& smooth_meanv,
    xt::xarray<double>& smooth_stda,
    xt::xarray<double>& smooth_meana
)
{
    smooth_stdv = this->Smooth_Std(filter_v_target, smooth_window);
    smooth_meanv = this->Smooth_Mean(filter_v_target, smooth_window);
    smooth_stda = this->Smooth_Std(filter_a_target, smooth_window);
    smooth_meana = this->Smooth_Mean(filter_a_target, smooth_window);
}

void Feature_Store::compute_curvature_features(
    const xt::xarray<double>& smooth_meanv,
    const xt::xarray<double>& smooth_meana,
    const xt::xarray<double>& smooth_stdv,
    const xt::xarray<double>& smooth_stda,
    std::vector<double>& features
)
{
    features.push_back(this->Curvature(this->Filter_V->row_element(0), this->Filter_a->row_element(0)));
    features.push_back(this->Curvature(smooth_meanv, smooth_meana));
    features.push_back(this->Curvature(smooth_stdv, smooth_stda));
    features.push_back(this->Curvature(smooth_stdv, smooth_meana));
    features.push_back(this->Curvature(smooth_meanv, smooth_stda));
}

void Feature_Store::compute_similarity_features(
    const xt::xarray<double>& smooth_meanv,
    const xt::xarray<double>& smooth_meana,
    const xt::xarray<double>& smooth_stdv,
    const xt::xarray<double>& smooth_stda,
    std::vector<double>& features
)
{
    features.push_back(this->Similarity(this->Filter_V->row_element(0), this->Filter_a->row_element(0)));
    features.push_back(this->Similarity(smooth_meanv, smooth_meana));
    features.push_back(this->Similarity(smooth_stdv, smooth_stda));
    features.push_back(this->Similarity(smooth_stdv, smooth_meana));
    features.push_back(this->Similarity(smooth_meanv, smooth_stda));
}

void Feature_Store::compute_angle_features(
    const xt::xarray<double>& filter_x_target,
    const xt::xarray<double>& filter_v_target,
    const xt::xarray<double>& filter_a_target,
    std::vector<double>& features
)
{
    // 方位角
    features.push_back(this->CalAzimuth(filter_x_target));
    features.push_back(this->CalAzimuth(filter_v_target));
    features.push_back(this->CalAzimuth(filter_a_target));
    
    // 仰角
    features.push_back(this->CalElevation(filter_x_target));
    features.push_back(this->CalElevation(filter_v_target));
    features.push_back(this->CalElevation(filter_a_target));
}

std::vector<double> Feature_Store::get_trace_features(int smooth_window)
{
    std::vector<double> features;
    features.reserve(37);  // 预分配空间，与Python版本特征数量一致
    // 添加基础目标系特征
    for (int i = 0; i < 3; i++) {
        features.push_back(this->Filter_x_Target->row_element(0)[i]);
    };
    for (int i = 0; i < 3; i++) {
        features.push_back(this->Filter_V_Target->row_element(0)[i]);
    };
    for (int i = 0; i < 3; i++) {
        features.push_back(this->Filter_a_Target->row_element(0)[i]);
    };
    //2. 计算平滑特征
    xt::xarray<double> smooth_stdv, smooth_meanv, smooth_stda, smooth_meana;
    compute_smooth_features(
      smooth_window, 
      this->Filter_V_Target->row_element(0), 
      this->Filter_a_Target->row_element(0),
      smooth_stdv, 
      smooth_meanv, 
      smooth_stda, 
      smooth_meana
    );
    for (int i = 0; i < 3; i++) {
        features.push_back(smooth_stdv[i]);
    };
    for (int i = 0; i < 3; i++) {
        features.push_back(smooth_meanv[i]);
    };
    for (int i = 0; i < 3; i++) {
        features.push_back(smooth_stda[i]);
    };
    for (int i = 0; i < 3; i++) {
        features.push_back(smooth_meana[i]);
    };

    //3. 计算曲率特征
    compute_curvature_features(smooth_meanv, smooth_meana, smooth_stdv, smooth_stda, features);

    //4. 计算相似度特征
    compute_similarity_features(smooth_meanv, smooth_meana, smooth_stdv, smooth_stda, features);

    //5. 计算角度特征
    compute_angle_features(
      this->Filter_x_Target->row_element(0),
      this->Filter_V_Target->row_element(0),
      this->Filter_a_Target->row_element(0),
      features
    );
  

    return features;
}



double Feature_Store::Dif(
    double value1,
    double value2,
    int window,
    double deltaT=0.08
)
{
    return  (value1-value2)/window/deltaT;
}


double Feature_Store::Modu(
  const xt::xarray<double>& row_vector
)
{
   return sqrt(row_vector[0]*row_vector[0]+row_vector[1]*row_vector[1]+row_vector[2]*row_vector[2]);
 
}


double Feature_Store::CalElevation(
  const xt::xarray<double>& row_vector
)
{
  double gdj,h,R;
  h =row_vector[2];
  R = sqrt(row_vector[0]*row_vector[0]+row_vector[1]*row_vector[1]);
  if ((R<EPSILON)&&(R>-EPSILON))
  {
    if ((h<EPSILON)&&(h>-EPSILON))
    {
      gdj = 0.0;
    }
    else
    {
      if (h>EPSILON)
      {
        gdj = 1500.0/RADTOMIL;
      }
      else
      {
        gdj = -1500.0/RADTOMIL;
      }
    }
  }
  else
  {
    gdj = atan(h/R);
  }
  return gdj;
}


double Feature_Store::CalAzimuth(
  const xt::xarray<double>& row_vector
)
{
  double nAzimuth = 0.0;
  double x = row_vector[0];
  double y = row_vector[1];
  if ((x<EPSILON) && (x>-EPSILON))
  {
    if((y<EPSILON)&&(y>-EPSILON))
    {
      nAzimuth = 0.0;
    }
    else
    {
      if(y >0.0)
      {
        nAzimuth = 1500.0/RADTOMIL;
      }
      else
      {
        nAzimuth = 4500.0/RADTOMIL;
      }
    }
  }
  else
  {
    if(x>0.0)
    {
      if ((y<EPSILON)&&(y>-EPSILON))
      {
        nAzimuth = 0.0;
      }
      else
      {
        if(y>0.0)
        {
          nAzimuth = atan(y/x);
        }
        else
        {
          nAzimuth = atan(y/x)+6000.0/RADTOMIL;
        }
      }
    }
    else
    {
      if((y<EPSILON)&&(y>-EPSILON))
      {
        nAzimuth = 3000.0/RADTOMIL;
      }
      else
      {
        nAzimuth = atan(y/x)+3000.0/RADTOMIL;
      }
    }
  }
  return nAzimuth;
}

double Feature_Store::Similarity(
  const xt::xarray<double>& row_vector1,
  const xt::xarray<double>& row_vector2
)
{
  double result = row_vector1[0]*row_vector2[0]+row_vector1[1]*row_vector2[1]+row_vector1[2]*row_vector2[2];
  return result/(Modu(row_vector1)+EPSILON)/(Modu(row_vector2)+EPSILON);
}

double Feature_Store::Curvature(
  const xt::xarray<double>& rowVel,
  const xt::xarray<double>& rowAcc
)
{
  xt::xarray<double> cross_dot = {rowVel[1]*rowAcc[2]-rowVel[2]*rowAcc[1],
                                  rowVel[2]*rowAcc[0]-rowVel[0]*rowAcc[2],
                                  rowVel[0]*rowAcc[1]-rowVel[1]*rowAcc[0]};
  double result =  Modu(cross_dot)/(pow(Modu(rowVel),3.0)+EPSILON);
  return result;
}


xt::xarray<double> Feature_Store::Mul(
    const xt::xarray<double>&  row_vector,
    double multi_rate
)
{
    xt::xarray<double>  result = {
      row_vector[0]*multi_rate,
      row_vector[1]*multi_rate,
      row_vector[2]*multi_rate
        };
    return result;
}

xt::xarray<double> Feature_Store::Add(
    const xt::xarray<double>&  row_vector1,
    const xt::xarray<double>&  row_vector2
)
{
    xt::xarray<double>  result = {
      row_vector1[0] + row_vector2[0],
      row_vector1[1] + row_vector2[1],
      row_vector1[2] + row_vector2[2]
        };
    return result;
}

xt::xarray<double> Feature_Store::Sub(
    const xt::xarray<double>&  row_vector1,
    const xt::xarray<double>&  row_vector2)
{
    xt::xarray<double>  result = {
      row_vector1[0] - row_vector2[0],
      row_vector1[1] - row_vector2[1],
      row_vector1[2] - row_vector2[2]
        };
    return result;
}


xt::xarray<double> Feature_Store::Dif(
    const xt::xarray<double>&  matrix,
    int window,
    double deltaT=0.08
)
{
    xt::xarray<double>  result = {
        Dif(matrix(0,0),matrix(window,0),window,deltaT),
        Dif(matrix(0,1),matrix(window,1),window,deltaT),
        Dif(matrix(0,2),matrix(window,2),window,deltaT)
        };
    return result;
}


xt::xarray<double> Feature_Store::Smooth_Mean(
    const xt::xarray<double>&  matrix,
    int window
)
{
    return xt::mean(xt::view(matrix,xt::range(0,window),xt::all()),{0});
}

xt::xarray<double> Feature_Store::Smooth_Std(
    const xt::xarray<double>&  matrix,
    int window
)
{
    return xt::stddev(xt::view(matrix,xt::range(0,window),xt::all()),{0});
}


//vector2vector Operator 
xt::xarray<double> Feature_Store::Real2Target(
  const xt::xarray<double>& real_row_vector,
  const xt::xarray<double>& base_row_vector
)
{
  double beta_h = CalAzimuth(base_row_vector);
  double beta_l = CalElevation(base_row_vector);
  double Tx = cos(beta_h)*cos(beta_l)*real_row_vector[0]+\
              sin(beta_h)*cos(beta_l)*real_row_vector[1]+\
              sin(beta_l)*real_row_vector[2];
  double Ty = -sin(beta_h)*real_row_vector[0]+\
              cos(beta_h)*real_row_vector[1];
  double Tz = -cos(beta_h)*sin(beta_l)*real_row_vector[0]+\
              -sin(beta_h)*sin(beta_l)*real_row_vector[1]+\
              cos(beta_l)*real_row_vector[2];     
  xt::xarray<double> result = {Tx,Ty,Tz};
  return result;
}

xt::xarray<double> Feature_Store::Target2Real(
  const xt::xarray<double>& target_row_vector,
  const xt::xarray<double>& base_row_vector
)
{
  double beta_h = CalAzimuth(base_row_vector);
  double beta_l = CalElevation(base_row_vector);
  double Tx = cos(beta_h)*cos(beta_l)*target_row_vector[0]+\
              -sin(beta_h)*target_row_vector[1]+\
              -sin(beta_l)*cos(beta_h)*target_row_vector[2];
  double Ty = sin(beta_h)*cos(beta_l)*target_row_vector[0]+\
              cos(beta_h)*target_row_vector[1]+\
              -sin(beta_h)*sin(beta_l)*target_row_vector[2];
  double Tz = sin(beta_l)*target_row_vector[0]+\
              0+\
              cos(beta_l)*target_row_vector[2];     
  xt::xarray<double> result = {Tx,Ty,Tz};
  return result;
}

void Feature_Store::update_image(const std::vector<unsigned char >& new_image_data) {
    image_data = new_image_data;
    image_initialized = true;
}

const std::vector<unsigned char >& Feature_Store::get_image_data() const {
    return image_data;
}



