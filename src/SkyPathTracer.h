#ifndef SKY_PATH_TRACER
#define SKY_PATH_TRACER

#if __cplusplus
extern "C" {
#endif

#define SKY_SPECTRUM_N 8

struct skyPathTracerParams {
    int ozone_absorption;
    int shadow_steps;
    float base_density;
    float sun_strength;
    int steps;
    int phase_function;
    float mie_g;
    float mie_diameter;
    float density_rayleigh;
    float density_mie;
    float density_ozone;
    float rayleigh_height_falloff;
    float mie_height_falloff;
    float wavelength_red;
    float wavelength_green;
    float wavelength_blue;
    float carbondioxide_percent;
    int use_ms;
    float ground_visibility;
    float ms_factor;
    int convertSpectrum;
    int use_tm_lut;
    int ground;
    float ground_albedo;
    int use_static_sun_solid_angle;
    float ozone_layer_thickness;
    int uniform_wavelengths;
    float wavelengths[SKY_SPECTRUM_N];
    int use_tracking;
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