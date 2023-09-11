class erosplus
{
private:

    cv::Mat surface;
    cv::Mat true_surface;
    int k_size{7};
    double absorb{0.05};
    double inject{0.01};
    double balance{100.0};

public:
    void init(int width, int height, int kernel_size = 7, double absorb = 0.05, double inject = 0.003)
    {
        k_size = kernel_size % 2 == 0 ? kernel_size + 1 : kernel_size;
        this->absorb = absorb;
        this->inject = inject;
        surface = cv::Mat(height + k_size-1, width + k_size-1, CV_64F, cv::Scalar(10));
        true_surface = surface({k_size/2, k_size/2, width, height});
    }

    void update(int u, int v)
    {
        static int half_kernel = k_size / 2;
        static cv::Rect region = {0, 0, k_size, k_size};
        static cv::Mat region_mat;
        static double nabsorb = 1.0 - absorb;
        
        region.x = u; region.y = v; region_mat = surface(region);
        double& c = surface.at<double>(v+half_kernel, u+half_kernel);
        if(c > balance*2.0) {
            region_mat *= 0.5;
        } else {
            double ca = 0.0;
            for(int x = 0; x < k_size; x++) {
                for(int y = 0; y < k_size; y++) {
                    double& cc = region_mat.at<double>(y, x);
                    ca += cc * absorb;
                    cc *= nabsorb;
                }
            }
            c += ca + balance*inject;
        }
    }

    cv::Mat& getSurface()
    {
        static cv::Mat ret;
        true_surface.convertTo(ret, CV_8U);
        return ret;
    }

};
