#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <thread>
#include <fstream>
#include <chrono>
#include <random>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <omp.h>
#include <atomic>

// 1. Vec2 类定义
class Vec2 {
public:
    float x, y;
    Vec2(float x = 0, float y = 0) : x(x), y(y) {}
};

// 2. Vec3 类定义
class Vec3 {
public:
    float x, y, z;

    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float scalar) const { return Vec3(x * scalar, y * scalar, z * scalar); }
    Vec3 operator/(float scalar) const { return Vec3(x / scalar, y / scalar, z / scalar); }
    Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    Vec3 cross(const Vec3& v) const {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    Vec3 normalize() const {
        float len = std::sqrt(x * x + y * y + z * z);
        return Vec3(x / len, y / len, z / len);
    }
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    float& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    const float& operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    // 添加友元运算符
    friend Vec3 operator*(float t, const Vec3& v) {
        return Vec3(t * v.x, t * v.y, t * v.z);
    }

    Vec3 multiply(const Vec3& v) const {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }
};

// 3. Ray 类定义
class Ray {
public:
    Vec3 orig, dir;
    Ray(const Vec3& orig = Vec3(), const Vec3& dir = Vec3()) : orig(orig), dir(dir.normalize()) {}
};

// 4. AABB 类定义
class AABB {
public:
    Vec3 min, max;

    AABB(const Vec3& min = Vec3(), const Vec3& max = Vec3()) : min(min), max(max) {}

    float surfaceArea() const {
        Vec3 d = max - min;
        return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
    }

    bool intersect(const Ray& ray, float& tMin, float& tMax) const {
        for (int i = 0; i < 3; ++i) {
            float invD = 1.0f / (ray.dir[i] != 0.0f ? ray.dir[i] : 1e-6);
            float t0 = (min[i] - ray.orig[i]) * invD;
            float t1 = (max[i] - ray.orig[i]) * invD;
            if (invD < 0.0f) std::swap(t0, t1);
            tMin = std::max(tMin, t0);
            tMax = std::min(tMax, t1);
            if (tMax <= tMin) return false;
        }
        return true;
    }
};

// 5. 辅助函数定义
float random_float() {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

float random_float(float min, float max) {
    return min + (max - min) * random_float();
}

Vec3 randomUnitVector() {
    while (true) {
        Vec3 p(random_float() * 2.0f - 1.0f, 
               random_float() * 2.0f - 1.0f, 
               random_float() * 2.0f - 1.0f);
        if (p.dot(p) < 1) return p.normalize();
    }
}

Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - n * (2 * v.dot(n));
}

Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = std::min(-uv.dot(n), 1.0f);
    Vec3 r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    Vec3 r_out_parallel = n * (-std::sqrt(std::abs(1.0f - r_out_perp.dot(r_out_perp))));
    return r_out_perp + r_out_parallel;
}

AABB unionAABB(const AABB& a, const AABB& b) {
    Vec3 min(std::min(a.min.x, b.min.x),
             std::min(a.min.y, b.min.y),
             std::min(a.min.z, b.min.z));
    Vec3 max(std::max(a.max.x, b.max.x),
             std::max(a.max.y, b.max.y),
             std::max(a.max.z, b.max.z));
    return AABB(min, max);
}

void updateProgress(float progress) {
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

// 6. Material 类定义
class Material {
public:
    virtual ~Material() = default;
    virtual bool scatter(const Ray& ray_in, const Vec3& hit_point,
                        const Vec3& normal, Ray& scattered,
                        Vec3& attenuation) const = 0;
};

// 7. Geometry 类定义
class Geometry {
public:
    Material* material;
    virtual ~Geometry() = default;
    virtual bool intersect(const Ray& ray, float& t, Vec3& normal) const = 0;
    virtual AABB getBoundingBox() const = 0;
};

// 8. 具体材质类定义
class Lambertian : public Material {
public:
    Vec3 albedo;
    float roughness;

    Lambertian(const Vec3& a, float r = 1.0f) 
        : albedo(a), roughness(r) {}

    bool scatter(const Ray& ray_in, const Vec3& hit_point,
                const Vec3& normal, Ray& scattered,
                Vec3& attenuation) const override {
        Vec3 scatter_direction;
        if (roughness >= 1.0f) {
            // 完全漫反射
            scatter_direction = normal + randomUnitVector();
        } else {
            // 有光泽的表面
            Vec3 perfect_reflect = reflect(ray_in.dir.normalize(), normal);
            scatter_direction = perfect_reflect + randomUnitVector() * roughness;
        }
        
        if (scatter_direction.dot(normal) < 0) {
            scatter_direction = normal;
        }
        
        scattered = Ray(hit_point + normal * 0.001f, scatter_direction.normalize());
        attenuation = albedo;
        return true;
    }
};

class Metal : public Material {
public:
    Vec3 albedo;
    float fuzz;

    Metal(const Vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    bool scatter(const Ray& ray_in, const Vec3& hit_point,
                const Vec3& normal, Ray& scattered,
                Vec3& attenuation) const override {
        Vec3 reflected = reflect(ray_in.dir.normalize(), normal);
        scattered = Ray(hit_point, reflected + randomUnitVector() * fuzz);
        attenuation = albedo;
        return scattered.dir.dot(normal) > 0;
    }
};

class Dielectric : public Material {
public:
    float ior;  // 折射率

    Dielectric(float index_of_refraction) : ior(index_of_refraction) {}

    bool scatter(const Ray& ray_in, const Vec3& hit_point,
                const Vec3& normal, Ray& scattered,
                Vec3& attenuation) const override {
        attenuation = Vec3(1.0f);
        float refraction_ratio = ray_in.dir.dot(normal) < 0 ? 1.0f/ior : ior;

        Vec3 unit_direction = ray_in.dir.normalize();
        float cos_theta = std::min(-unit_direction.dot(normal), 1.0f);
        float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float())
            direction = reflect(unit_direction, normal);
        else
            direction = refract(unit_direction, normal, refraction_ratio);

        scattered = Ray(hit_point, direction);
        return true;
    }

private:
    static float reflectance(float cosine, float ref_idx) {
        float r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};

// 9. Sphere 类定义
class Sphere : public Geometry {
public:
    Vec3 center;
    float radius;

    Sphere(const Vec3& center, float radius, Material* mat) 
        : center(center), radius(radius) {
        this->material = mat;
    }

    bool intersect(const Ray& ray, float& t, Vec3& normal) const override {
        Vec3 oc = ray.orig - center;
        float a = ray.dir.dot(ray.dir);
        float b = 2.0f * oc.dot(ray.dir);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;

        float sqrtd = std::sqrt(discriminant);
        float root = (-b - sqrtd) / (2.0f * a);
        if (root < 0.001f) {
            root = (-b + sqrtd) / (2.0f * a);
            if (root < 0.001f) return false;
        }

        t = root;
        Vec3 hit_point = ray.orig + ray.dir * t;
        normal = (hit_point - center) / radius;
        return true;
    }

    AABB getBoundingBox() const override {
        Vec3 min = center - Vec3(radius, radius, radius);
        Vec3 max = center + Vec3(radius, radius, radius);
        return AABB(min, max);
    }
};

// 10. BVH 相关类定义
struct BVHNode {
    AABB bounds;
    std::vector<Geometry*> objects;
    BVHNode* left = nullptr;
    BVHNode* right = nullptr;
};

class BVH {
public:
    BVHNode* root;

    BVH(const std::vector<Geometry*>& objects) {
        // 创建一个可修改的副本
        std::vector<Geometry*> buildObjects = objects;
        root = build(buildObjects, 0, buildObjects.size());
    }

    ~BVH() {
        deleteBVH(root);
    }

    bool intersect(const Ray& ray, float& t, Vec3& normal, Material*& material) const {
        return intersectNode(root, ray, t, normal, material);
    }

private:
    void deleteBVH(BVHNode* node) {
        if (!node) return;
        deleteBVH(node->left);
        deleteBVH(node->right);
        delete node;
    }

    BVHNode* build(std::vector<Geometry*>& objects, size_t start, size_t end) {
        BVHNode* node = new BVHNode();

        if (end - start <= 2) {
            node->objects.assign(objects.begin() + start, objects.begin() + end);
            node->bounds = computeBoundingBox(node->objects);
            return node;
        }

        // 计算所有对象的包围盒中心点的包围盒
        AABB centroidBounds;
        for (size_t i = start; i < end; i++) {
            AABB box = objects[i]->getBoundingBox();
            Vec3 centroid = (box.min + box.max) * 0.5f;
            centroidBounds = unionAABB(centroidBounds, AABB(centroid, centroid));
        }

        // 选择最长轴
        Vec3 extent = centroidBounds.max - centroidBounds.min;
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent[axis]) axis = 2;

        // 根据选定轴对对象进行分区
        size_t mid = (start + end) / 2;
        auto compareFunc = [axis](Geometry* a, Geometry* b) {
            AABB box_a = a->getBoundingBox();
            AABB box_b = b->getBoundingBox();
            return (box_a.min[axis] + box_a.max[axis]) <
                   (box_b.min[axis] + box_b.max[axis]);
        };

        std::nth_element(objects.begin() + start,
                        objects.begin() + mid,
                        objects.begin() + end,
                        compareFunc);

        node->left = build(objects, start, mid);
        node->right = build(objects, mid, end);
        node->bounds = unionAABB(node->left->bounds, node->right->bounds);

        return node;
    }

    bool intersectNode(BVHNode* node, const Ray& ray, float& t, Vec3& normal, Material*& material) const {
        float tMin = 0.001f, tMax = t;
        if (!node->bounds.intersect(ray, tMin, tMax)) return false;

        if (!node->objects.empty()) {
            bool hit = false;
            for (const auto& object : node->objects) {
                float temp_t = t;
                Vec3 temp_normal;
                if (object->intersect(ray, temp_t, temp_normal)) {
                    if (temp_t < t) {
                        t = temp_t;
                        normal = temp_normal;
                        material = object->material;
                        hit = true;
                    }
                }
            }
            return hit;
        }

        bool hit_left = node->left && intersectNode(node->left, ray, t, normal, material);
        bool hit_right = node->right && intersectNode(node->right, ray, t, normal, material);
        return hit_left || hit_right;
    }

    AABB computeBoundingBox(const std::vector<Geometry*>& objects) {
        if (objects.empty()) return AABB();
        AABB bounds = objects[0]->getBoundingBox();
        for (size_t i = 1; i < objects.size(); i++) {
            bounds = unionAABB(bounds, objects[i]->getBoundingBox());
        }
        return bounds;
    }
};

// 11. Camera 类定义
class Camera {
public:
    Vec3 position;
    Vec3 forward;
    Vec3 up;
    Vec3 right;
    float fov;
    float aspect;
    float aperture;
    float focusDistance;

    Camera(const Vec3& pos, const Vec3& target, const Vec3& up_dir,
           float fov_degrees, float aspect_ratio,
           float aperture, float focus_dist)
        : position(pos), fov(fov_degrees * M_PI / 180.0f),
          aspect(aspect_ratio), aperture(aperture),
          focusDistance(focus_dist) {
        forward = (target - position).normalize();
        right = forward.cross(up_dir).normalize();
        up = right.cross(forward);
    }

    Ray generateRay(float s, float t) const {
        Vec3 rd = randomInUnitDisk() * (aperture / 2.0f);
        Vec3 offset = right * rd.x + up * rd.y;

        float half_height = std::tan(fov / 2.0f);
        float half_width = aspect * half_height;

        Vec3 dir = forward +
                  right * ((2.0f * s - 1.0f) * half_width) +
                  up * ((2.0f * (1.0f - t) - 1.0f) * half_height);

        return Ray(position + offset,
                  (position + dir * focusDistance - (position + offset)).normalize());
    }

private:
    Vec3 randomInUnitDisk() const {
        while (true) {
            Vec3 p(random_float() * 2.0f - 1.0f,
                  random_float() * 2.0f - 1.0f,
                  0);
            if (p.dot(p) < 1) return p;
        }
    }
};

// 添加光源类
class Light {
public:
    Vec3 position;
    Vec3 color;
    float radius;

    Light(const Vec3& pos, const Vec3& col, float r) 
        : position(pos), color(col), radius(r) {}

    Vec3 samplePoint() const {
        // 在圆盘上随机采样
        float r = sqrt(random_float());
        float theta = 2 * M_PI * random_float();
        Vec3 offset(r * cos(theta), 0, r * sin(theta));
        return position + offset * radius;
    }
};

// 修改 PathTrace 函数
Vec3 pathTrace(const Ray& ray, const BVH& bvh, const std::vector<Light>& lights, int depth) {
    if (depth <= 0) return Vec3(0);

    float t = std::numeric_limits<float>::max();
    Vec3 normal;
    Material* material = nullptr;
    
    if (bvh.intersect(ray, t, normal, material)) {
        Vec3 hit_point = ray.orig + ray.dir * t;
        
        // 直接光照
        Vec3 direct_light(0);
        for (const auto& light : lights) {
            // 对光源采样
            Vec3 light_point = light.samplePoint();
            Vec3 light_dir = (light_point - hit_point).normalize();
            float light_distance = (light_point - hit_point).length();
            
            // 阴影检测
            Ray shadow_ray(hit_point + normal * 0.001f, light_dir);
            float shadow_t = std::numeric_limits<float>::max();
            Vec3 shadow_normal;
            Material* shadow_material = nullptr;
            
            bool is_shadowed = bvh.intersect(shadow_ray, shadow_t, shadow_normal, shadow_material) 
                              && shadow_t < light_distance;
            
            if (!is_shadowed) {
                float cos_theta = std::max(0.0f, normal.dot(light_dir));
                float attenuation = 1.0f / (light_distance * light_distance);
                direct_light = direct_light + light.color * cos_theta * attenuation;
            }
        }

        // 间接光照
        Ray scattered;
        Vec3 attenuation;
        
        // 俄罗斯轮盘赌
        if (depth > 3) {
            float rr_prob = 0.8f;
            if (random_float() > rr_prob) {
                return direct_light;
            }
            attenuation = attenuation / rr_prob;
        }

        if (material && material->scatter(ray, hit_point, normal, scattered, attenuation)) {
            Vec3 indirect_light = pathTrace(scattered, bvh, lights, depth - 1);
            return direct_light + attenuation * indirect_light;
        }
        
        return direct_light;
    }

    // 环境光
    float skyBlend = 0.5f * (ray.dir.y + 1.0f);
    return Vec3(1.0f) * (1.0f - skyBlend) + Vec3(0.5f, 0.7f, 1.0f) * skyBlend;
}

// 13. 图像保存函数
void saveImage(const std::vector<std::vector<Vec3>>& framebuffer, const std::string& folderName) {
    std::filesystem::create_directories(folderName);
    
    int height = framebuffer.size();
    int width = framebuffer[0].size();
    
    cv::Mat image(height, width, CV_8UC3);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const Vec3& color = framebuffer[y][x];
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uchar>(color.z * 255),
                static_cast<uchar>(color.y * 255),
                static_cast<uchar>(color.x * 255)
            );
        }
    }
    
    std::string imagePath = folderName + "/render.png";
    cv::imwrite(imagePath, image);
    
    std::string sourcePath = folderName + "/raytracer_copy.cpp";
    std::filesystem::copy_file("raytracer.cpp", sourcePath, 
                             std::filesystem::copy_options::overwrite_existing);
    
    std::cout << "Image saved as: " << imagePath << std::endl;
    std::cout << "Source code saved as: " << sourcePath << std::endl;
}

// 14. std::swap 特化
namespace std {
    template<>
    void swap<Geometry*>(Geometry*& a, Geometry*& b) noexcept {
        Geometry* temp = a;
        a = b;
        b = temp;
    }
}

// 在其他类定义之后添加
class Triangle : public Geometry {
public:
    Vec3 v0, v1, v2;  // 三个顶点
    Vec3 n0, n1, n2;  // 顶点法线
    Vec2 t0, t1, t2;  // 纹理坐标

    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2,
            const Vec3& n0, const Vec3& n1, const Vec3& n2,
            Material* mat) : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2) {
        material = mat;
    }

    bool intersect(const Ray& ray, float& t, Vec3& normal) const override {
        Vec3 edge1 = v1 - v0;
        Vec3 edge2 = v2 - v0;
        Vec3 h = ray.dir.cross(edge2);
        float a = edge1.dot(h);

        if (std::abs(a) < 1e-6f) return false;

        float f = 1.0f / a;
        Vec3 s = ray.orig - v0;
        float u = f * s.dot(h);

        if (u < 0.0f || u > 1.0f) return false;

        Vec3 q = s.cross(edge1);
        float v = f * ray.dir.dot(q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float temp_t = f * edge2.dot(q);
        if (temp_t > 1e-6f) {
            t = temp_t;
            normal = ((1.0f-u-v) * n0) + (u * n1) + (v * n2);
            normal = normal.normalize();
            return true;
        }
        return false;
    }

    AABB getBoundingBox() const override {
        Vec3 min(
            std::min({v0.x, v1.x, v2.x}),
            std::min({v0.y, v1.y, v2.y}),
            std::min({v0.z, v1.z, v2.z})
        );
        Vec3 max(
            std::max({v0.x, v1.x, v2.x}),
            std::max({v0.y, v1.y, v2.y}),
            std::max({v0.z, v1.z, v2.z})
        );
        return AABB(min, max);
    }
};

// 将 PixelStats 类定义移到这里（在 hybridRender 函数之前）
class PixelStats {
public:
    Vec3 mean;
    Vec3 M2;
    int n = 0;

    void update(const Vec3& x) {
        n++;
        Vec3 delta = x - mean;
        mean = mean + delta / n;
        Vec3 delta2 = x - mean;
        M2 = M2 + delta.multiply(delta2);
    }

    Vec3 variance() const {
        if (n < 2) return Vec3(std::numeric_limits<float>::infinity());
        return M2 / (n - 1);
    }
};

// 在 PixelStats 类之后，hybridRender 之前添加 rasterize 函数的声明
void rasterize(const std::vector<Triangle>& triangles,
              std::vector<std::vector<Vec3>>& framebuffer,
              std::vector<std::vector<float>>& zbuffer,
              const Camera& camera);

// 然后是 hybridRender 函数
void hybridRender(const std::vector<Triangle>& triangles,
                 const std::vector<Geometry*>& rayTracingObjects,
                 std::vector<std::vector<Vec3>>& framebuffer,
                 const Camera& camera,
                 const BVH& bvh,
                 const std::vector<Light>& lights,
                 int samples,
                 int maxDepth,
                 float variance_threshold = 0.01f) {
    int width = framebuffer[0].size();
    int height = framebuffer.size();

    // 用于光栅化的Z-buffer
    std::vector<std::vector<float>> zbuffer(
        height, std::vector<float>(width, std::numeric_limits<float>::infinity())
    );

    // 创建像素统计数组
    std::vector<std::vector<PixelStats>> pixel_stats(height, 
        std::vector<PixelStats>(width));

    // 1. 首先进行光栅化渲染
    rasterize(triangles, framebuffer, zbuffer, camera);

    // 2. 然后对每个像素进行路径追踪，但要考虑Z-buffer
    std::atomic<int> pixel_count{0};
    const int total_pixels = width * height;

    #pragma omp parallel for collapse(2) schedule(dynamic, 16)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3 rasterColor = framebuffer[y][x];
            Vec3 rtColor(0, 0, 0);
            
            // 路径追踪
            for (int s = 0; s < samples; s++) {
                float u = (x + random_float()) / width;
                float v = 1.0f - (y + random_float()) / height;
                
                Ray ray = camera.generateRay(u, v);
                Vec3 sample_color = pathTrace(ray, bvh, lights, maxDepth);
                rtColor = rtColor + sample_color;
                
                // 更新像素统计
                pixel_stats[y][x].update(sample_color);

                // 检查方差
                if (s % 16 == 0 && s >= 32) {
                    Vec3 var = pixel_stats[y][x].variance();
                    float max_var = std::max({var.x, var.y, var.z});
                    if (max_var < variance_threshold) {
                        break;  // 提前结束采样
                    }
                }
            }
            
            rtColor = rtColor / samples;
            rtColor = Vec3(std::sqrt(rtColor.x), std::sqrt(rtColor.y), std::sqrt(rtColor.z));

            // 混合光栅化和路径追踪结果
            float zDepth = zbuffer[y][x];
            if (zDepth == std::numeric_limits<float>::infinity()) {
                framebuffer[y][x] = rtColor;
            } else {
                float blend = std::exp(-zDepth * 0.1f);
                framebuffer[y][x] = rasterColor * (1.0f - blend) + rtColor * blend;
            }

            // 更新进度
            if (pixel_count.fetch_add(1) % 64 == 0) {
                #pragma omp critical
                {
                    float progress = float(pixel_count) / total_pixels;
                    updateProgress(progress);
                }
            }
        }
    }
}

// 最后是 rasterize 函数的实现
void rasterize(const std::vector<Triangle>& triangles,
              std::vector<std::vector<Vec3>>& framebuffer,
              std::vector<std::vector<float>>& zbuffer,
              const Camera& camera) {
    int width = framebuffer[0].size();
    int height = framebuffer.size();

    // 初始化z-buffer
    zbuffer = std::vector<std::vector<float>>(
        height, std::vector<float>(width, std::numeric_limits<float>::infinity())
    );

    // 视口变换矩阵
    auto viewport = [&](const Vec3& p) -> Vec3 {
        float x = (p.x + 1.0f) * width / 2.0f;
        float y = (1.0f - p.y) * height / 2.0f;
        return Vec3(x, y, p.z);
    };

    // 遍历所有三角形
    for (const auto& triangle : triangles) {
        // 简单的视口变换
        Vec3 p0 = viewport(triangle.v0);
        Vec3 p1 = viewport(triangle.v1);
        Vec3 p2 = viewport(triangle.v2);

        // 计算包围盒
        int minX = std::max(0, int(std::min({p0.x, p1.x, p2.x})));
        int maxX = std::min(width - 1, int(std::max({p0.x, p1.x, p2.x})));
        int minY = std::max(0, int(std::min({p0.y, p1.y, p2.y})));
        int maxY = std::min(height - 1, int(std::max({p0.y, p1.y, p2.y})));

        // 遍历包围盒中的像素
        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                Vec3 p(x + 0.5f, y + 0.5f, 0);
                
                // 计算重心坐标
                float area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
                float w0 = ((p.x - p1.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p.y - p1.y)) / area;
                float w1 = ((p.x - p2.x) * (p0.y - p2.y) - (p0.x - p2.x) * (p.y - p2.y)) / area;
                float w2 = 1.0f - w0 - w1;

                // 检查点是否在三角形内
                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    // 计算z值
                    float z = w0 * p0.z + w1 * p1.z + w2 * p2.z;
                    
                    // z-buffer测试
                    if (z < zbuffer[y][x]) {
                        zbuffer[y][x] = z;
                        
                        // 插值法线
                        Vec3 normal = (w0 * triangle.n0 + w1 * triangle.n1 + w2 * triangle.n2).normalize();
                        
                        // 简单的光照计算
                        Vec3 lightDir = Vec3(0, 1, 1).normalize();
                        float intensity = std::max(0.0f, normal.dot(lightDir));
                        
                        // 设置像素颜色
                        framebuffer[y][x] = Vec3(intensity);
                    }
                }
            }
        }
    }
}

// 15. 主函数
int main() {
    // 设置OpenMP线程数
    int num_threads = std::thread::hardware_concurrency(); // 获取CPU核心数
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " threads for rendering.\n";

    // 初始化场景
    std::vector<Geometry*> sceneObjects;
    
    // 添加材质
    Material* redMetal = new Metal(Vec3(0.8f, 0.2f, 0.2f), 0.1f);
    Material* blueMetal = new Metal(Vec3(0.2f, 0.2f, 0.8f), 0.1f);
    Material* whiteDiffuse = new Lambertian(Vec3(0.8f, 0.8f, 0.8f));
    Material* glass = new Dielectric(1.5f);
    Material* goldMetal = new Metal(Vec3(0.8f, 0.6f, 0.2f), 0.2f);
    
    // 添加球体
    sceneObjects.push_back(new Sphere(Vec3(0, 1, 0), 1.0f, glass));
    sceneObjects.push_back(new Sphere(Vec3(-4, 1, 0), 1.0f, redMetal));
    sceneObjects.push_back(new Sphere(Vec3(4, 1, 0), 1.0f, goldMetal));
    
    // 添加地面
    sceneObjects.push_back(new Sphere(Vec3(0, -1000, 0), 1000.0f, whiteDiffuse));
    
    // 添加小球
    for(int a = -11; a < 11; a++) {
        for(int b = -11; b < 11; b++) {
            float choose_mat = random_float();
            Vec3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

            if ((center - Vec3(4, 0.2f, 0)).length() > 0.9f) {
                if (choose_mat < 0.8f) {
                    // 漫反射
                    Vec3 albedo = Vec3(random_float(), random_float(), random_float());
                    sceneObjects.push_back(new Sphere(center, 0.2f, 
                        new Lambertian(albedo)));
                } else if (choose_mat < 0.95f) {
                    // 金属
                    Vec3 albedo = Vec3(0.5f * (1 + random_float()),
                                     0.5f * (1 + random_float()),
                                     0.5f * (1 + random_float()));
                    float fuzz = 0.5f * random_float();
                    sceneObjects.push_back(new Sphere(center, 0.2f,
                        new Metal(albedo, fuzz)));
                } else {
                    // 玻璃
                    sceneObjects.push_back(new Sphere(center, 0.2f,
                        new Dielectric(1.5f)));
                }
            }
        }
    }

    // 构建BVH
    BVH bvh(sceneObjects);

    // 渲染参数
    int width = 800, height = 600;
    const int samples = 4;  // 每个像素的采样数
    const int maxDepth = 5; // 光线反弹的最大深度
    std::vector<std::vector<Vec3>> framebuffer(height, std::vector<Vec3>(width));

    auto startTime = std::chrono::high_resolution_clock::now();

    // 创建相机
    Vec3 lookfrom(13, 2, 3);
    Vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0f;
    float aperture = 0.1f;
    Camera camera(lookfrom, lookat, Vec3(0,1,0), 20.0f, float(width)/height,
                 aperture, dist_to_focus);

    // 添加原子计数器
    std::atomic<int> pixel_count{0};
    const int total_pixels = width * height;

    // 添加三角形网格
    std::vector<Triangle> triangles;
    
    // 添加一个简单的三角形地面
    Vec3 n(0, 1, 0);  // 向上的法线
    triangles.push_back(Triangle(
        Vec3(-10, -0.5f, -10), Vec3(-10, -0.5f, 10), Vec3(10, -0.5f, -10),
        n, n, n,
        whiteDiffuse
    ));
    triangles.push_back(Triangle(
        Vec3(10, -0.5f, -10), Vec3(-10, -0.5f, 10), Vec3(10, -0.5f, 10),
        n, n, n,
        whiteDiffuse
    ));

    // 添加光源
    std::vector<Light> lights;
    lights.push_back(Light(Vec3(0, 5, 0), Vec3(1.0f, 0.9f, 0.8f) * 5.0f, 2.0f));  // 主光源
    lights.push_back(Light(Vec3(-5, 3, -5), Vec3(0.3f, 0.4f, 0.9f) * 3.0f, 1.0f)); // 补光

    // 执行混合渲染
    hybridRender(triangles, sceneObjects, framebuffer, camera, bvh, 
                lights, samples, maxDepth);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Rendering completed in " << duration << "ms.\n";

    // 生成时间戳文件夹名
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "renders/" << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
    std::string folderName = ss.str();

    // 保存图像和源代码
    saveImage(framebuffer, folderName);

    // 清理内存
    for (auto obj : sceneObjects) delete obj;
    delete redMetal;
    delete blueMetal;
    delete whiteDiffuse;
    delete glass;
    delete goldMetal;

    return 0;
}
