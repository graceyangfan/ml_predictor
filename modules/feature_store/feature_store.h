#ifndef FEATURE_STORE_H 
#define FEATURE_STORE_H
#include <vector>
#include "batch_vector.h" 
#define RADTOMIL 954.9296585513
#define EPSILON 0.0000001

class Feature_Store
{
    public:
        BatchVector*  Observe;        // 观测数据
        BatchVector*  Filter_P;       // 滤波位置
        BatchVector*  Filter_V;       // 滤波速度
        BatchVector*  Filter_a;       // 滤波加速度
        BatchVector*  Base_Vector;    // 基准向量
        BatchVector*  Filter_x_Target; // 目标系位置
        BatchVector*  Filter_V_Target; // 目标系速度
        BatchVector*  Filter_a_Target; // 目标系加速度
        
        double deltaT;                // 时间间隔
        int based_window;             // 基准窗口大小
        int cache_length;             // 缓存长度
        std::vector<unsigned char> image_data; // 图像数据

    private:
        bool track_initialized = false;  // 航迹特征是否初始化
        bool image_initialized = false;  // 图像是否初始化
            
        void compute_smooth_features(int smooth_window,
            const xt::xarray<double>& filter_v_target,
            const xt::xarray<double>& filter_a_target,
            xt::xarray<double>& smooth_stdv,
            xt::xarray<double>& smooth_meanv,
            xt::xarray<double>& smooth_stda,
            xt::xarray<double>& smooth_meana);
            
        void compute_curvature_features(
            const xt::xarray<double>& smooth_meanv,
            const xt::xarray<double>& smooth_meana,
            const xt::xarray<double>& smooth_stdv,
            const xt::xarray<double>& smooth_stda,
            std::vector<double>& features);
            
        void compute_similarity_features(
            const xt::xarray<double>& smooth_meanv,
            const xt::xarray<double>& smooth_meana,
            const xt::xarray<double>& smooth_stdv,
            const xt::xarray<double>& smooth_stda,
            std::vector<double>& features);
            
        void compute_angle_features(
            const xt::xarray<double>& filter_x_target,
            const xt::xarray<double>& filter_v_target,
            const xt::xarray<double>& filter_a_target,
            std::vector<double>& features);

    public:
        Feature_Store(
            double deltaT,
            int based_window,
            int cache_length
        );
        ~Feature_Store();

        void update(
            double Observe_x, double Observe_y, double Observe_z,
            double Filter_P_x, double Filter_P_y, double Filter_P_z,
            double Filter_V_x, double Filter_V_y, double Filter_V_z,
            double Filter_a_x, double Filter_a_y, double Filter_a_z
        );

        bool is_track_initialized() const;
        bool is_image_initialized() const;
        bool is_fully_initialized() const;

        // 获取特征向量，使用与Python相同的特征构建逻辑
        std::vector<double> get_trace_features(int smooth_window = 5);

        // 基础计算函数
        double Dif(double value1, double value2, int window, double deltaT);
        double Modu(const xt::xarray<double>& row_vector);
        double CalElevation(const xt::xarray<double>& row_vector);
        double CalAzimuth(const xt::xarray<double>& row_vector);
        double Similarity(const xt::xarray<double>& row_vector1, const xt::xarray<double>& row_vector2);
        double Curvature(const xt::xarray<double>& rowVel, const xt::xarray<double>& rowAcc);

        // 向量操作函数
        xt::xarray<double> Mul(const xt::xarray<double>& row_vector, double multi_rate);
        xt::xarray<double> Add(const xt::xarray<double>& row_vector1, const xt::xarray<double>& row_vector2);
        xt::xarray<double> Sub(const xt::xarray<double>& row_vector1, const xt::xarray<double>& row_vector2);
        xt::xarray<double> Dif(const xt::xarray<double>& matrix, int window,double deltaT);
        xt::xarray<double> Smooth_Mean(const xt::xarray<double>& matrix, int window);
        xt::xarray<double> Smooth_Std(const xt::xarray<double>& matrix, int window);
        xt::xarray<double> Real2Target(const xt::xarray<double>& real_row_vector, const xt::xarray<double>& base_row_vector);
        xt::xarray<double> Target2Real(const xt::xarray<double>& target_row_vector, const xt::xarray<double>& base_row_vector);

        // 图像数据相关函数
        void update_image(const std::vector<unsigned char>& new_image_data);
        const std::vector<unsigned char>& get_image_data() const;
};
#endif 