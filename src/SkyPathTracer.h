#ifndef SKY_PATH_TRACER
#define SKY_PATH_TRACER

#if __cplusplus
extern "C" {
#endif

struct skyPathTracerParams {
    int ozone_absorption;
    int shadow_steps;
    float base_density;
    float sun_strength;
    int steps;
    int use_cs_mie;
    float mie_g;
} typedef skyPathTracerParams;

void renderPathTracer(  const skyPathTracerParams        model,
                        const double                     albedo,
                        const double                     altitude,
                        const double                     azimuth,
                        const double                     elevation,
                        const int                        resolution,
                        const int                        view,
                        const double                     visibility,
                        float**                          outResult);

#if __cplusplus
}
#endif

#endif /* SKY_PATH_TRACER */