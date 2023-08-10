#include "SkyPathTracer.h"

#include "SkyOzoneCrossSections.h"
#include "SkySpectrum2RGB.h"
#include "SkySunRadiance.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define __device__ static

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Defines Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef eps
#define eps 0.001f
#endif /* eps */

#ifndef PI
#define PI 3.14159265359f
#endif /* PI */

#define SKY_EARTH_RADIUS 6371.0f
#define SKY_SUN_RADIUS 696340.0f
#define SKY_SUN_DISTANCE 149597870.0f
#define SKY_MOON_RADIUS 1737.4f
#define SKY_MOON_DISTANCE 384399.0f
#define SKY_ATMO_HEIGHT 100.0f
#define SKY_ATMO_RADIUS (SKY_ATMO_HEIGHT + SKY_EARTH_RADIUS)
#define SKY_HEIGHT_OFFSET 0.0005f

//#define SKY_RAYLEIGH_SCATTERING get_color(5.8f * 0.001f, 13.558f * 0.001f, 33.1f * 0.001f)
#define SKY_RAYLEIGH_SCATTERING INT_PARAMS.rayleigh_scattering
#define SKY_MIE_SCATTERING get_color(3.996f * 0.001f, 3.996f * 0.001f, 3.996f * 0.001f)
#define SKY_OZONE_SCATTERING 0.0f

#define SKY_RAYLEIGH_EXTINCTION SKY_RAYLEIGH_SCATTERING
#define SKY_MIE_EXTINCTION get_color(4.440f * 0.001f, 4.440f * 0.001f, 4.440f * 0.001f)
//#define SKY_OZONE_EXTINCTION get_color(0.65f * 0.001f, 1.881f * 0.001f, 0.085f * 0.001f)
#define SKY_OZONE_EXTINCTION INT_PARAMS.ozone_absorption

#define SKY_RAYLEIGH_DISTRIBUTION (1.0f / 8.0f)
#define SKY_MIE_DISTRIBUTION (1.0f / 1.2f)

#define SKY_MS_TEX_SIZE 32
#define SKY_TM_TEX_WIDTH 256
#define SKY_TM_TEX_HEIGHT 64

#define SKY_USE_WIDE_SPECTRUM 1

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Math Helper Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

float fromUnitToSubUvs(float u, float resolution) { return (u + 0.5f / resolution) * (resolution / (resolution + 1.0f)); }
float fromSubUvsToUnit(float u, float resolution) { return (u - 0.5f / resolution) * (resolution / (resolution - 1.0f)); }

#define __saturatef(x) fmaxf(0.0f, fminf(1.0f, (x)))

struct float2 {
    float x;
    float y;
} typedef float2;

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

#if SKY_USE_WIDE_SPECTRUM
struct RGBF {
  float v[SKY_SPECTRUM_N];
} typedef RGBF;
struct sRGBF {
  float r;
  float g;
  float b;
} typedef sRGBF;
#else
struct RGBF {
  float r;
  float g;
  float b;
} typedef RGBF;
#define sRGBF RGBF
#endif



__device__ float2 make_float2(const float a, const float b) {
    float2 result;

    result.x = a;
    result.y = b;

    return result;
}

__device__ vec3 get_vector(const float x, const float y, const float z) {
  vec3 result;

  result.x = x;
  result.y = y;
  result.z = z;

  return result;
}

__device__ vec3 add_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;

  return result;
}

__device__ vec3 sub_vector(const vec3 a, const vec3 b) {
  vec3 result;

  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;

  return result;
}

__device__ vec3 scale_vector(vec3 vector, const float scale) {
  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

__device__ float dot_product(const vec3 a, const vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float get_length(const vec3 vector) {
  return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

#define rnorm3df(x,y,z) (1.0f / (sqrtf((x) * (x) + (y) * (y) + (z) * (z))))

__device__ vec3 normalize_vector(vec3 vector) {
  const float scale = rnorm3df(vector.x, vector.y, vector.z);

  vector.x *= scale;
  vector.y *= scale;
  vector.z *= scale;

  return vector;
}

#if SKY_USE_WIDE_SPECTRUM
__device__ RGBF get_color(const float r, const float g, const float b) {
  RGBF result;

  // For wide spectrum we only allow a broadcast all operation
  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = r;
  }

  return result;
}

__device__ RGBF add_color(const RGBF a, const RGBF b) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = a.v[i] + b.v[i];
  }

  return result;
}

__device__ RGBF sub_color(const RGBF a, const RGBF b) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = a.v[i] - b.v[i];
  }

  return result;
}

__device__ RGBF scale_color(const RGBF a, const float b) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = a.v[i] * b;
  }

  return result;
}

__device__ RGBF mul_color(const RGBF a, const RGBF b) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = a.v[i] * b.v[i];
  }

  return result;
}

__device__ RGBF inv_color(const RGBF a) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = 1.0f / a.v[i];
  }

  return result;
}

__device__ RGBF exp_color(const RGBF a) {
  RGBF result;

  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    result.v[i] = expf(a.v[i]);
  }

  return result;
}
#else
__device__ RGBF get_color(const float r, const float g, const float b) {
  RGBF result;

  result.r = r;
  result.g = g;
  result.b = b;

  return result;
}

__device__ RGBF add_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r + b.r;
  result.g = a.g + b.g;
  result.b = a.b + b.b;

  return result;
}

__device__ RGBF sub_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r - b.r;
  result.g = a.g - b.g;
  result.b = a.b - b.b;

  return result;
}

__device__ RGBF scale_color(const RGBF a, const float b) {
  RGBF result;

  result.r = a.r * b;
  result.g = a.g * b;
  result.b = a.b * b;

  return result;
}

__device__ RGBF mul_color(const RGBF a, const RGBF b) {
  RGBF result;

  result.r = a.r * b.r;
  result.g = a.g * b.g;
  result.b = a.b * b.b;

  return result;
}

__device__ RGBF inv_color(const RGBF a) {
  RGBF result;

  result.r = 1.0f / a.r;
  result.g = 1.0f / a.g;
  result.b = 1.0f / a.b;

  return result;
}

__device__ RGBF exp_color(const RGBF a) {
  RGBF result;

  result.r = expf(a.r);
  result.g = expf(a.g);
  result.b = expf(a.b);

  return result;
}
#endif



/*
 * Computes the distance to the first intersection of a ray with a sphere. To check for any hit use sphere_ray_hit.
 * @param ray Ray direction.
 * @param origin Ray origin.
 * @param p Center of the sphere.
 * @param r Radius of the sphere.
 * @result Value a such that origin + a * ray is a point on the sphere.
 */
__device__ float sphere_ray_intersection(const vec3 ray, const vec3 origin, const vec3 p, const float r) {
  const vec3 diff = sub_vector(origin, p);
  const float dot = dot_product(diff, ray);
  const float r2  = r * r;
  const float a   = dot_product(ray, ray);
  const float c   = dot_product(diff, diff) - r2;
  const vec3 k    = sub_vector(diff, scale_vector(ray, dot));
  const float d   = 4.0f * a * (r2 - dot_product(k, k));

  if (d < 0.0f)
    return FLT_MAX;

  const vec3 h   = add_vector(diff, scale_vector(ray, -dot / a));
  const float sd = sqrtf(a * (r2 - dot_product(h, h)));
  const float q  = -dot + copysignf(1.0f, -dot) * sd;

  const float t0 = c / q;

  if (t0 >= 0.0f)
    return t0;

  const float t1 = q / a;
  return (t1 < 0.0f) ? FLT_MAX : t1;
}

/*
 * Computes whether a ray hits a sphere with (0,0,0) as its center. To compute the distance see sph_ray_int_p0.
 * @param ray Ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result 1 if the ray hits the sphere, 0 else.
 */
__device__ int sph_ray_hit_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const float a   = dot_product(ray, ray);
  const float c   = dot_product(origin, origin) - r2;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = 4.0f * a * (r2 - dot_product(k, k));

  if (d < 0.0f)
    return 0;

  const vec3 h   = add_vector(origin, scale_vector(ray, -dot / a));
  const float sd = sqrtf(a * (r2 - dot_product(h, h)));
  const float q  = -dot + copysignf(1.0f, -dot) * sd;

  const float t0 = c / q;

  return (t0 >= 0.0f);
}

/*
 * Computes the distance to the first intersection of a ray with a sphere with (0,0,0) as its center.
 * @param ray Ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result Value a such that origin + a * ray is a point on the sphere.
 */
__device__ float sph_ray_int_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const float a   = dot_product(ray, ray);
  const float c   = dot_product(origin, origin) - r2;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = 4.0f * a * (r2 - dot_product(k, k));

  if (d < 0.0f)
    return FLT_MAX;

  const vec3 h   = add_vector(origin, scale_vector(ray, -dot / a));
  const float sd = sqrtf(a * (r2 - dot_product(h, h)));
  const float q  = -dot + copysignf(1.0f, -dot) * sd;

  const float t0 = c / q;

  if (t0 >= 0.0f)
    return t0;

  const float t1 = q / a;
  return (t1 < 0.0f) ? FLT_MAX : t1;
}


/*
 * Computes the distance to the last intersection of a ray with a sphere with (0,0,0) as its center.
 * @param ray Ray direction.
 * @param origin Ray origin.
 * @param r Radius of the sphere.
 * @result Value a such that origin + a * ray is a point on the sphere.
 */
__device__ float sph_ray_int_back_p0(const vec3 ray, const vec3 origin, const float r) {
  const float dot = dot_product(origin, ray);
  const float r2  = r * r;
  const float a   = dot_product(ray, ray);
  const float c   = dot_product(origin, origin) - r2;
  const vec3 k    = sub_vector(origin, scale_vector(ray, dot));
  const float d   = 4.0f * a * (r2 - dot_product(k, k));

  if (d < 0.0f)
    return FLT_MAX;

  const vec3 h   = add_vector(origin, scale_vector(ray, -dot / a));
  const float sd = sqrtf(a * (r2 - dot_product(h, h)));
  const float q  = -dot + copysignf(1.0f, -dot) * sd;

  const float t1 = q / a;

  if (t1 >= 0.0f)
    return t1;

  const float t0 = c / q;
  return (t0 < 0.0f) ? FLT_MAX : t0;
}

/*
 * Computes solid angle when sampling a sphere.
 * @param p Center of sphere.
 * @param r Radius of sphere.
 * @param origin Point from which you sample.
 * @param normal Normalized normal of surface from which you sample.
 * @result Solid angle of sphere.
 */
__device__ float sample_sphere_solid_angle(const vec3 p, const float r, const vec3 origin) {
  vec3 dir      = sub_vector(p, origin);
  const float d = get_length(dir);

  if (d < r)
    return 2.0f * PI;

  const float a = asinf(r / d);

  return 2.0f * PI * a * a;
}

__device__ float cornette_shanks(const float cos_angle, const float g) {
	return (3.0f * (1.0f - g * g) * (1.0f + cos_angle * cos_angle)) / (4.0f * PI * 2.0f * (2.0f + g * g) * pow(1.0f + g * g - 2.0f * g * cos_angle, 3.0f/2.0f));
}

__device__ float henvey_greenstein(const float cos_angle, const float g) {
  return (1.0f - g * g) / (4.0f * PI * powf(1.0f + g * g - 2.0f * g * cos_angle, 1.5f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Additional Structs Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

struct skyInternalParams {
  RGBF sun_color;
  vec3 sun_pos;
  RGBF rayleigh_scattering;
  RGBF ozone_absorption;
  RGBF* ms_lut;
  RGBF* tm_lut;
#if SKY_USE_WIDE_SPECTRUM
  RGBF wavelengths;
#endif
} typedef skyInternalParams;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Sky Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

static skyPathTracerParams PARAMS;
static skyInternalParams INT_PARAMS;

// Hillaire2020
__device__ float2 sky_transmittance_lut_uv(float height, float zenith_cos_angle) {
  height += SKY_EARTH_RADIUS;
	float H = sqrtf(fmaxf(0.0f, SKY_ATMO_RADIUS * SKY_ATMO_RADIUS - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS));
	float rho = sqrtf(fmaxf(0.0f, height * height - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS));

	float discriminant = height * height * (zenith_cos_angle * zenith_cos_angle - 1.0f) + SKY_ATMO_RADIUS * SKY_ATMO_RADIUS;
	float d = fmaxf(0.0f, (-height * zenith_cos_angle + sqrtf(discriminant))); // Distance to atmosphere boundary

	float d_min = SKY_ATMO_RADIUS - height;
	float d_max = rho + H;
	float x_mu = (d - d_min) / (d_max - d_min);
	float x_r = rho / H;

	return make_float2(x_mu, x_r);
}

__device__ RGBF sky_sample_tex(RGBF* tex, float2 uv, int width, int height) {
    float ms_x = fmaxf(0.0f, fminf(1.0f, uv.x));
    float ms_y = fmaxf(0.0f, fminf(1.0f, uv.y));

    ms_x = fromUnitToSubUvs(ms_x, width);
    ms_y = fromUnitToSubUvs(ms_y, height);

    ms_x *= (width - 1);
    ms_y *= (height - 1);

    const int ms_ix = (int)(ms_x);
    const int ms_iy = (int)(ms_y);

    const int ms_ix1 = ms_ix + 1;
    const int ms_iy1 = ms_iy + 1;

    const RGBF ms00 = tex[ms_ix + ms_iy * width];
    const RGBF ms01 = tex[ms_ix + ms_iy1 * width];
    const RGBF ms10 = tex[ms_ix1 + ms_iy * width];
    const RGBF ms11 = tex[ms_ix1 + ms_iy1 * width];

    const float w00 = (ms_ix1 - ms_x) * (ms_iy1 - ms_y);
    const float w01 = (ms_ix1 - ms_x) * (ms_y - ms_iy);
    const float w10 = (ms_x - ms_ix) * (ms_iy1 - ms_y);
    const float w11 = (ms_x - ms_ix) * (ms_y - ms_iy);

    RGBF interp = scale_color(ms00, w00);
    interp = add_color(interp, scale_color(ms01, w01));
    interp = add_color(interp, scale_color(ms10, w10));
    interp = add_color(interp, scale_color(ms11, w11));

    return interp;
}

__device__ float sky_rayleigh_phase(const float cos_angle) {
  return 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);
}

__device__ float sky_mie_phase(const float cos_angle) {
  if (PARAMS.use_cs_mie) {
    return cornette_shanks(cos_angle, PARAMS.mie_g);
  } else {
    return henvey_greenstein(cos_angle, PARAMS.mie_g);
  }
}

__device__ float sky_rayleigh_density(const float height) {
  return 2.5f * PARAMS.base_density * expf(-height * (1.0f / PARAMS.rayleigh_height_falloff));
}

__device__ float sky_mie_density(const float height) {
  // INSO (insoluble = dust-like particles)
  float INSO = expf(-height * (1.0f / PARAMS.mie_height_falloff));
  /*if (height < 2.0f) {
    INSO = expf(-height * (1.0f / 8.0f));
  } else if (height < 12.0f) {
    INSO = 0.3f * expf(-height * (1.0f / 8.0f));
  }*/

  // WASO (water soluble = biogenic particles, organic carbon)
  float WASO = 0.0f;
  if (height < 2.0f) {
    WASO = 1.0f + 0.125f * (2.0f - height);
  } else if (height < 3.0f) {
    WASO = 3.0f - height;
  }
  WASO *= 60.0f / PARAMS.ground_visibility;

  return PARAMS.base_density * (INSO + WASO);
}

__device__ float sky_ozone_density(const float height) {
  if (!PARAMS.ozone_absorption)
    return 0.0f;

#if 0
  if (height > 20.0f) {
    return PARAMS.base_density * expf(-0.07f * (height - 20.0f))/*fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.04f)*/;
  }
  else {
    return PARAMS.base_density * fmaxf(0.1f, 1.0f - (20.0f - height) * 0.066666667f);
  }
#elif 1
  const float min_val = (height > 20.0f) ? 0.0f : 0.1f;
  return PARAMS.base_density * fmaxf(min_val, 1.0f - fabsf(height - 25.0f) / PARAMS.ozone_layer_thickness);
#else
  return PARAMS.base_density * expf(-height * (1.0f / PARAMS.rayleigh_height_falloff));
#endif
}

__device__ float sky_height(const vec3 point) {
  return get_length(point) - SKY_EARTH_RADIUS;
}

__device__ RGBF sky_extinction(const vec3 origin, const vec3 ray, const float start, const float length) {
  if (length <= 0.0f)
    return get_color(0.0f, 0.0f, 0.0f);

  const int steps       = PARAMS.shadow_steps;
  float step_size       = length / steps;
  RGBF density          = get_color(0.0f, 0.0f, 0.0f);
  float reach           = start;

  for (float i = 0; i < steps; i += 1.0f) {
    const float newReach = start + length * (i + 0.3f) / steps;
    step_size = newReach - reach;
    reach = newReach;

    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height           = sky_height(pos);

    const float density_rayleigh = sky_rayleigh_density(height) * PARAMS.density_rayleigh;
    const float density_mie      = sky_mie_density(height) * PARAMS.density_mie;
    const float density_ozone    = sky_ozone_density(height) * PARAMS.density_ozone;

    RGBF D = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    D      = add_color(D, scale_color(SKY_MIE_EXTINCTION, density_mie));
    D      = add_color(D, scale_color(SKY_OZONE_EXTINCTION, density_ozone));

    density = add_color(density, scale_color(D, step_size));
  }

  density = scale_color(density, -1.0f);

  return exp_color(density);
}

/*
 * Computes the start and length of a ray path through atmosphere.
 * @param origin Start point of ray in sky space.
 * @param ray Direction of ray
 * @result 2 floats, first value is the start, second value is the length of the path.
 */
__device__ float2 sky_compute_path(const vec3 origin, const vec3 ray, const float min_height, const float max_height) {
  const float height = get_length(origin);

  if (height <= min_height)
    return make_float2(0.0f, -FLT_MAX);

  float distance;
  float start = 0.0f;
  if (height > max_height) {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    const float atmo_dist2 = sph_ray_int_back_p0(ray, origin, max_height);

    distance = fminf(earth_dist - atmo_dist, atmo_dist2 - atmo_dist);
    start    = atmo_dist;
  }
  else {
    const float earth_dist = sph_ray_int_p0(ray, origin, min_height);
    const float atmo_dist  = sph_ray_int_p0(ray, origin, max_height);
    distance               = fminf(earth_dist, atmo_dist);
  }

  return make_float2(start, distance);
}

// assumes that wavelengths are sorted with blue lowest, red highest
static float sky_interpolate_radiance_at_wavelength(RGBF radiance, float wavelength) {
#if SKY_USE_WIDE_SPECTRUM
  if (wavelength < INT_PARAMS.wavelengths.v[0]) {
    return radiance.v[0];
  }

  for (int i = 1; i < SKY_SPECTRUM_N; i++) {
    if (wavelength < INT_PARAMS.wavelengths.v[i]) {
      float u = (wavelength - INT_PARAMS.wavelengths.v[i - 1]) / (INT_PARAMS.wavelengths.v[i] - INT_PARAMS.wavelengths.v[i - 1]);
      return radiance.v[i] * u + radiance.v[i - 1] * (1.0f - u);
    }
  }

  return radiance.v[SKY_SPECTRUM_N - 1];
#else
  if (wavelength < PARAMS.wavelength_blue) {
    return radiance.b;
  }

  if (wavelength < PARAMS.wavelength_green) {
    float u = (wavelength - PARAMS.wavelength_blue) / (PARAMS.wavelength_green - PARAMS.wavelength_blue);
    return radiance.g * u + radiance.b * (1.0f - u);
  }

  if (wavelength < PARAMS.wavelength_red) {
    float u = (wavelength - PARAMS.wavelength_green) / (PARAMS.wavelength_red - PARAMS.wavelength_green);
    return radiance.r * u + radiance.g * (1.0f - u);
  }

  return radiance.r;
#endif

}

// Conversion of spectrum to sRGB as in Bruneton2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
__device__ sRGBF sky_convert_wavelengths_to_sRGB(RGBF radiance) {
#if !SKY_USE_WIDE_SPECTRUM
  if (!PARAMS.convertSpectrum) {
    return radiance;
  }
#endif

  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;

#if SKY_USE_WIDE_SPECTRUM
  const float step_size = INT_PARAMS.wavelengths.v[1] - INT_PARAMS.wavelengths.v[0];
  for (int i = 0; i < SKY_SPECTRUM_N; i++) {
    x += CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 1) * radiance.v[i];
    y += CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 2) * radiance.v[i];
    z += CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 3) * radiance.v[i];
  }
#else
  float step_size = 1.0f;
  for (float lambda = SKY_WAVELENGTH_MIN; lambda < SKY_WAVELENGTH_MAX; lambda += step_size) {
    const float v = sky_interpolate_radiance_at_wavelength(radiance, lambda);
    x += CieColorMatchingFunctionTableValue(lambda, 1) * v;
    y += CieColorMatchingFunctionTableValue(lambda, 2) * v;
    z += CieColorMatchingFunctionTableValue(lambda, 3) * v;
  }
  x *= /*683.0f*/ step_size / (SKY_WAVELENGTH_MAX - SKY_WAVELENGTH_MIN);
  y *= /*683.0f*/ step_size / (SKY_WAVELENGTH_MAX - SKY_WAVELENGTH_MIN);
  z *= /*683.0f*/ step_size / (SKY_WAVELENGTH_MAX - SKY_WAVELENGTH_MIN);
#endif

  sRGBF result;
  result.r = 3.2406f * x - 1.5372f * y - 0.4986f * z;
  result.g = -0.9689f * x + 1.8758f * y + 0.0415f * z;
  result.b = 0.0557f * x - 0.2040f * y + 1.0570f * z;

  return result;
}

__device__ sRGBF sky_compute_atmosphere(const vec3 origin, const vec3 ray, const float limit) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX) {
    return sky_convert_wavelengths_to_sRGB(result);
  }

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  RGBF transmittance = get_color(1.0f, 1.0f, 1.0f);

  if (distance > 0.0f) {
    const int steps       = PARAMS.steps;
    float step_size       = distance / steps;
    float reach           = start;

    const RGBF sun_radiance = scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength);

    float light_angle;

    if (PARAMS.use_static_sun_solid_angle) {
      light_angle = sample_sphere_solid_angle(INT_PARAMS.sun_pos, SKY_SUN_RADIUS, origin);
    }

    for (float i = 0; i < steps; i += 1.0f) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size = newReach - reach;
      reach = newReach;

      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float height = sky_height(pos);

      const vec3 ray_scatter  = normalize_vector(sub_vector(INT_PARAMS.sun_pos, pos));
      const float cos_angle = dot_product(ray, ray_scatter);
      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle);

      const float shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;

      RGBF extinction_sun;
      if (PARAMS.use_tm_lut) {
        const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);
        const float2 tm_uv = sky_transmittance_lut_uv(height, zenith_cos_angle);
        extinction_sun = sky_sample_tex(INT_PARAMS.tm_lut, tm_uv, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT);
      } else {
        const float scatter_distance = sph_ray_int_p0(ray_scatter, pos, SKY_ATMO_RADIUS);
        extinction_sun = sky_extinction(pos, ray_scatter, 0.0f, scatter_distance);
      }

      const float density_rayleigh = sky_rayleigh_density(height) * PARAMS.density_rayleigh;
      const float density_mie      = sky_mie_density(height) * PARAMS.density_mie;
      const float density_ozone    = sky_ozone_density(height) * PARAMS.density_ozone;

      const RGBF scattering_rayleigh = scale_color(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const RGBF scattering_mie = scale_color(SKY_MIE_SCATTERING, density_mie);

      const RGBF extinction_rayleigh = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const RGBF extinction_mie = scale_color(SKY_MIE_EXTINCTION, density_mie);
      const RGBF extinction_ozone = scale_color(SKY_OZONE_EXTINCTION, density_ozone);

      const RGBF scattering = add_color(scattering_rayleigh, scattering_mie);
      const RGBF extinction = add_color(add_color(extinction_rayleigh, extinction_mie), extinction_ozone);
      const RGBF phaseTimesScattering = add_color(scale_color(scattering_rayleigh, phase_rayleigh), scale_color(scattering_mie, phase_mie));

      if (!PARAMS.use_static_sun_solid_angle) {
        light_angle = sample_sphere_solid_angle(INT_PARAMS.sun_pos, SKY_SUN_RADIUS, pos);
      }

      const RGBF ssRadiance = scale_color(mul_color(extinction_sun, phaseTimesScattering), shadow * light_angle);
      RGBF msRadiance = get_color(0.0f, 0.0f, 0.0f);

      if (PARAMS.use_ms) {
        const float sun_zenith_angle = dot_product(ray_scatter, normalize_vector(pos));
        const float2 ms_uv = make_float2(sun_zenith_angle * 0.5f + 0.5f, height / SKY_ATMO_HEIGHT);
        RGBF msTexResult = sky_sample_tex(INT_PARAMS.ms_lut, ms_uv, SKY_MS_TEX_SIZE, SKY_MS_TEX_SIZE);
        msRadiance = mul_color(msTexResult, scattering);
      }

      const RGBF S = mul_color(sun_radiance, add_color(ssRadiance, msRadiance));

      RGBF step_transmittance = extinction;
      step_transmittance = scale_color(step_transmittance, -step_size);
      step_transmittance = exp_color(step_transmittance);

      const RGBF Sint = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));

      result = add_color(result, mul_color(Sint, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);
    }
  }

  const float sun_hit   = sphere_ray_intersection(ray, origin, INT_PARAMS.sun_pos, SKY_SUN_RADIUS);
  const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);

  if (earth_hit > sun_hit) {
#if !SKY_USE_WIDE_SPECTRUM
    const vec3 sun_hit_pos  = add_vector(origin, scale_vector(ray, sun_hit));
    const float limb_factor = 1.0f + dot_product(normalize_vector(sub_vector(sun_hit_pos, INT_PARAMS.sun_pos)), ray);
    const float mu          = sqrtf(1.0f - limb_factor * limb_factor);

    const RGBF limb_color = get_color(0.397f, 0.503f, 0.652f);

    const RGBF limb_darkening = get_color(powf(mu, limb_color.r), powf(mu, limb_color.g), powf(mu, limb_color.b));

    RGBF S = mul_color(transmittance, scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength));
    S      = mul_color(S, limb_darkening);

    result = add_color(result, S);
#endif
  } else if (0 && PARAMS.ground && earth_hit < FLT_MAX) {
    // Turned off for single scattering because it looks ugly
    const vec3 earth_hit_pos  = add_vector(origin, scale_vector(ray, earth_hit));
    const float light_angle = sample_sphere_solid_angle(INT_PARAMS.sun_pos, SKY_SUN_RADIUS, earth_hit_pos);
    const vec3 ray_scatter  = normalize_vector(sub_vector(INT_PARAMS.sun_pos, earth_hit_pos));
    const float zenith_cos_angle = __saturatef(dot_product(normalize_vector(earth_hit_pos), ray_scatter));

    RGBF extinction_sun;
    if (PARAMS.use_tm_lut) {
      const float2 tm_uv = sky_transmittance_lut_uv(0.0f, zenith_cos_angle);
      extinction_sun = sky_sample_tex(INT_PARAMS.tm_lut, tm_uv, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT);
    } else {
      const float scatter_distance = sph_ray_int_p0(ray_scatter, earth_hit_pos, SKY_ATMO_RADIUS);
      extinction_sun = sky_extinction(earth_hit_pos, ray_scatter, 0.0f, scatter_distance);
    }

    result = add_color(result, scale_color(mul_color(scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength), mul_color(transmittance, extinction_sun)), PARAMS.ground_albedo * zenith_cos_angle * light_angle / PI));
  }

  return sky_convert_wavelengths_to_sRGB(result);
}

static vec3 angles_to_direction(const float altitude, const float azimuth) {
  vec3 dir;
  dir.x = cosf(azimuth) * cosf(altitude);
  dir.y = sinf(altitude);
  dir.z = sinf(azimuth) * cosf(altitude);

  return dir;
}

/// Computes direction corresponding to given pixel coordinates in up-facing or side-facing fisheye projection.
static vec3 pixelToDirection(int x, int y, int resolution, int view) {
    // Make circular image area in center of image.
    const double radius  = ((double)resolution) / 2.0;
    const double scaledx = (x + 0.5 - radius) / radius;
    const double scaledy = (y + 0.5 - radius) / radius;
    const double denom   = scaledx * scaledx + scaledy * scaledy + 1.0;

    if (denom > 2.0) {
        // Outside image area.
        return get_vector(0.0f, 0.0f, 0.0f);
    } else {
        // Stereographic mapping.
        if (view) {
            return get_vector(-2.0 * scaledy / denom, -2.0 * scaledx / denom, -(denom - 2.0) / denom);
        } else { // View::UpFacingFisheye
            return get_vector(2.0 * scaledx / denom, -(denom - 2.0) / denom, 2.0 * scaledy / denom);
        }
    }
}

// Wavelength in 1nm
static float computeRayleighScattering(const float wavelength) {
  // number molecules per cubic meter
  const double N = 2.546899 * 1e25;

  const double index = 1.000293;

  // wavelength in 1 meter
  const double lambda = ((double) wavelength) * 1e-9;

#if 0
  const double depolarisation = 0.035;
  const double F_air = (6.0 + 3.0 * depolarisation) / (6.0 - 7.0 * depolarisation);
#else
  // This formula needs wavelength to be in 1 micrometer
  const double lambda2 = lambda * 1e6 * lambda * 1e6;
  const double lambda4 = lambda2 * lambda2;

  // Bodhaine1999 (equation 23)
  const double F_argon = 1.0;
  const double F_carbondiox = 1.15;
  const double F_nitrogen = 1.024 + 3.17 * 1e-4 / lambda4;
  const double F_oxygen = 1.096 + 1.385 * 1e-3 / lambda2 + 1.448 * 1e-4 / lambda4;
  const double C_carbondiox = PARAMS.carbondioxide_percent;

  const double F_air = (78.084 * F_nitrogen + 20.946 * F_oxygen + 0.934 * F_argon + C_carbondiox * F_carbondiox)
                      / (78.084 + 20.946 + 0.934 + C_carbondiox);
#endif

  // Bodhaine1999 (equation 2)
  // Note the formula in the paper is per molecule
  const double scattering = (24.0 * PI * PI * PI * (index * index - 1.0) * (index * index - 1.0))
                          / (N * lambda * lambda * lambda * lambda * (index * index + 2.0) * (index * index + 2.0));

  // Convert from m^-1 to km^-1
  return (float) (scattering * F_air * 1e3);
}

static float computeOzoneAbsorption(const float wavelength) {
  // Convert from m^-1 to km^-1
  return (float) (ozoneCrossSectionSample(wavelength) * 1e3);
}

struct msScatteringResult {
  RGBF L;
  RGBF multiScatterAs1;
} typedef msScatteringResult;

static msScatteringResult computeMultiScatteringIntegration(const vec3 origin, const vec3 ray, const vec3 sun_pos) {
  msScatteringResult result;

  result.L = get_color(0.0f, 0.0f, 0.0f);
  result.multiScatterAs1 = get_color(0.0f, 0.0f, 0.0f);

  float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX) {
    return result;
  }

  const float start    = path.x;
  const float distance = path.y;

  if (distance > 0.0f) {
    const int steps       = 40;
    float step_size       = distance / steps;
    float reach           = start;

    const float light_angle = sample_sphere_solid_angle(sun_pos, SKY_SUN_RADIUS, origin);

    RGBF transmittance = get_color(1.0f, 1.0f, 1.0f);

    for (float i = 0; i < steps; i += 1.0f) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size = newReach - reach;
      reach = newReach;

      const vec3 pos = add_vector(origin, scale_vector(ray, reach));
      const float height = sky_height(pos);

      const vec3 ray_scatter  = normalize_vector(sub_vector(sun_pos, pos));
      const float cos_angle = dot_product(ray, ray_scatter);
      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle);

      RGBF extinction_sun;
      if (PARAMS.use_tm_lut) {
        const float zenith_cos_angle = dot_product(normalize_vector(pos), ray_scatter);
        const float2 tm_uv = sky_transmittance_lut_uv(height, zenith_cos_angle);
        extinction_sun = sky_sample_tex(INT_PARAMS.tm_lut, tm_uv, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT);
      } else {
        const float scatter_distance = sph_ray_int_p0(ray_scatter, pos, SKY_ATMO_RADIUS);
        extinction_sun = sky_extinction(pos, ray_scatter, 0.0f, scatter_distance);
      }

      const float density_rayleigh = sky_rayleigh_density(height) * PARAMS.density_rayleigh;
      const float density_mie      = sky_mie_density(height) * PARAMS.density_mie;
      const float density_ozone    = sky_ozone_density(height) * PARAMS.density_ozone;

      const RGBF scattering_rayleigh = scale_color(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const RGBF scattering_mie = scale_color(SKY_MIE_SCATTERING, density_mie);

      const RGBF extinction_rayleigh = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const RGBF extinction_mie = scale_color(SKY_MIE_EXTINCTION, density_mie);
      const RGBF extinction_ozone = scale_color(SKY_OZONE_EXTINCTION, density_ozone);

      const RGBF scattering = add_color(scattering_rayleigh, scattering_mie);
      const RGBF extinction = add_color(add_color(extinction_rayleigh, extinction_mie), extinction_ozone);
      const RGBF phaseTimesScattering = add_color(scale_color(scattering_rayleigh, phase_rayleigh), scale_color(scattering_mie, phase_mie));

      const float shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      const RGBF S = scale_color(mul_color(extinction_sun, phaseTimesScattering), shadow * light_angle);

      RGBF step_transmittance = extinction;
      step_transmittance = scale_color(step_transmittance, -step_size);
      step_transmittance = exp_color(step_transmittance);

      const RGBF ssInt = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));
      const RGBF msInt = mul_color(sub_color(scattering, mul_color(scattering, step_transmittance)), inv_color(extinction));

      result.L = add_color(result.L, mul_color(ssInt, transmittance));
      result.multiScatterAs1 = add_color(result.multiScatterAs1, mul_color(msInt, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);
    }

    const float sun_hit   = sphere_ray_intersection(ray, origin, sun_pos, SKY_SUN_RADIUS);
    const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);

    if (PARAMS.ground && earth_hit < sun_hit) {
      const vec3 earth_hit_pos  = add_vector(origin, scale_vector(ray, earth_hit));
      const float light_angle = sample_sphere_solid_angle(sun_pos, SKY_SUN_RADIUS, earth_hit_pos);
      const vec3 ray_scatter  = normalize_vector(sub_vector(sun_pos, earth_hit_pos));
      const float zenith_cos_angle = __saturatef(dot_product(normalize_vector(earth_hit_pos), ray_scatter));

      RGBF extinction_sun;
      if (PARAMS.use_tm_lut) {
        const float2 tm_uv = sky_transmittance_lut_uv(0.0f, zenith_cos_angle);
        extinction_sun = sky_sample_tex(INT_PARAMS.tm_lut, tm_uv, SKY_TM_TEX_WIDTH, SKY_TM_TEX_HEIGHT);
      } else {
        const float scatter_distance = sph_ray_int_p0(ray_scatter, earth_hit_pos, SKY_ATMO_RADIUS);
        extinction_sun = sky_extinction(earth_hit_pos, ray_scatter, 0.0f, scatter_distance);
      }

      result.L = add_color(result.L, scale_color(mul_color(transmittance, extinction_sun), PARAMS.ground_albedo * zenith_cos_angle * light_angle / PI));
    }
  }

  return result;
}

// Hillaire 2020
static void computeMultiScattering(RGBF** msTex) {
  *msTex = malloc(sizeof(RGBF) * SKY_MS_TEX_SIZE * SKY_MS_TEX_SIZE);

  #pragma omp parallel for
  for (int x = 0; x < SKY_MS_TEX_SIZE; x++) {
    for (int y = 0; y < SKY_MS_TEX_SIZE; y++) {
      float fx = ((float)x + 0.5f) / SKY_MS_TEX_SIZE;
      float fy = ((float)y + 0.5f) / SKY_MS_TEX_SIZE;

      fx = fromSubUvsToUnit(fx, SKY_MS_TEX_SIZE);
      fy = fromSubUvsToUnit(fy, SKY_MS_TEX_SIZE);

      float cos_angle = fx * 2.0f - 1.0f;
      vec3 sun_dir = get_vector(0.0f, cos_angle, sqrtf(__saturatef(1.0f - cos_angle * cos_angle)));
      float height = SKY_EARTH_RADIUS + __saturatef(fy + SKY_HEIGHT_OFFSET) * (SKY_ATMO_HEIGHT - SKY_HEIGHT_OFFSET);

      vec3 pos = get_vector(0.0f, height, 0.0f);

      RGBF inscatteredLuminance = get_color(0.0f, 0.0f, 0.0f);
      RGBF multiScattering = get_color(0.0f, 0.0f, 0.0f);

      const vec3 sun_pos = scale_vector(sun_dir, SKY_SUN_DISTANCE);

      const float sqrt_sample = 8.0f;

      for (int i = 0; i < 64; i++) {
        float a = 0.5f + i / 8;
        float b = 0.5f + (i - ((i/8) * 8));
        float randA = a / sqrt_sample;
        float randB = b / sqrt_sample;
        float theta = 2.0f * PI * randA;
        float phi = acosf(1.0f - 2.0f * randB) - 0.5f * PI;
        vec3 ray = angles_to_direction(phi, theta);

        msScatteringResult result = computeMultiScatteringIntegration(pos, ray, sun_pos);

        inscatteredLuminance = add_color(inscatteredLuminance, result.L);
        multiScattering = add_color(multiScattering, result.multiScatterAs1);
      }

      inscatteredLuminance = scale_color(inscatteredLuminance, 1.0f / (sqrt_sample * sqrt_sample));
      multiScattering = scale_color(multiScattering, 1.0f / (sqrt_sample * sqrt_sample));

      RGBF multiScatteringContribution = inv_color(sub_color(get_color(1.0f, 1.0f, 1.0f), multiScattering));

      RGBF L = scale_color(mul_color(inscatteredLuminance, multiScatteringContribution), PARAMS.ms_factor);

      (*msTex)[x + y * SKY_MS_TEX_SIZE] = L;
    }
  }
}

static RGBF computeTransmittanceOpticalDepth(float r, float mu) {
  const int steps = 500;

  // Distance to top of atmosphere
  const float disc = r * r * (mu * mu - 1.0f) + SKY_ATMO_RADIUS * SKY_ATMO_RADIUS;
  const float dist = fmaxf(-r * mu + sqrtf(fmaxf(0.0f, disc)), 0.0f);

  const float step_size = dist / steps;

  RGBF depth = get_color(0.0f, 0.0f, 0.0f);

  for (int i = 0; i <= steps; i++) {
    const float d_i = i * step_size;
    const float r_i = sqrtf(d_i * d_i + 2.0f * r * mu * d_i + r * r);

    const float density_rayleigh = sky_rayleigh_density(r_i - SKY_EARTH_RADIUS) * PARAMS.density_rayleigh;
    const float density_mie      = sky_mie_density(r_i - SKY_EARTH_RADIUS) * PARAMS.density_mie;
    const float density_ozone    = sky_ozone_density(r_i - SKY_EARTH_RADIUS) * PARAMS.density_ozone;

    const RGBF extinction_rayleigh = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    const RGBF extinction_mie = scale_color(SKY_MIE_EXTINCTION, density_mie);
    const RGBF extinction_ozone = scale_color(SKY_OZONE_EXTINCTION, density_ozone);

    const RGBF extinction = add_color(add_color(extinction_rayleigh, extinction_mie), extinction_ozone);

    const float w = (i == 0 || i == steps) ? 0.5f : 1.0f;

    depth = add_color(depth, scale_color(extinction, w * step_size));
  }

  return depth;
}

static void computeTransmittance(RGBF** tmTex) {
  *tmTex = malloc(sizeof(RGBF) * SKY_TM_TEX_WIDTH * SKY_TM_TEX_HEIGHT);

  #pragma omp parallel for
  for (int x = 0; x < SKY_TM_TEX_WIDTH; x++) {
    for (int y = 0; y < SKY_TM_TEX_HEIGHT; y++) {
      float fx = ((float)x + 0.5f) / SKY_TM_TEX_WIDTH;
      float fy = ((float)y + 0.5f) / SKY_TM_TEX_HEIGHT;

      fx = fromSubUvsToUnit(fx, SKY_TM_TEX_WIDTH);
      fy = fromSubUvsToUnit(fy, SKY_TM_TEX_HEIGHT);

      const float H = sqrtf(SKY_ATMO_RADIUS * SKY_ATMO_RADIUS - SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);
      const float rho = H * fy;
      const float r = sqrtf(rho * rho + SKY_EARTH_RADIUS * SKY_EARTH_RADIUS);

      const float d_min = SKY_ATMO_RADIUS - r;
      const float d_max = rho + H;
      const float d = d_min + fx * (d_max - d_min);

      float mu = (d == 0.0f) ? 1.0f : (H * H - rho * rho - d * d) / (2.0f * r * d);

      mu = fminf(1.0f, fmaxf(-1.0f, mu));

      RGBF optical_depth = computeTransmittanceOpticalDepth(r, mu);

      RGBF transmittance = exp_color(scale_color(optical_depth, -1.0f));

      (*tmTex)[x + y * SKY_TM_TEX_WIDTH] = transmittance;
    }
  }
}

void renderPathTracer(  const skyPathTracerParams        model,
                        const double                     albedo,
                        const double                     altitude,
                        const double                     azimuth,
                        const double                     elevation,
                        const int                        resolution,
                        const int                        view,
                        const double                     visibility,
                        float**                          outResult) {

  vec3 pos = {.x = 0.0f, .y = altitude / 1000.0f + SKY_EARTH_RADIUS + SKY_HEIGHT_OFFSET, .z = 0.0f};

  if (*outResult) {
    free(*outResult);
  }

  *outResult = malloc(sizeof(float) * resolution * resolution * 3);

  sRGBF* dst = (sRGBF*) *outResult;

  PARAMS = model;

  printf("//////////////////////////////////////\n");
  printf("//\n");
  printf("//  Path Tracer Data:\n");
  printf("//\n");
  {
    vec3 sun_pos = angles_to_direction(elevation, azimuth);
    sun_pos = normalize_vector(sun_pos);
    sun_pos = scale_vector(sun_pos, SKY_SUN_DISTANCE);
    sun_pos.y -= SKY_EARTH_RADIUS;

    INT_PARAMS.sun_pos = sun_pos;
#if SKY_USE_WIDE_SPECTRUM
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      INT_PARAMS.wavelengths.v[i] = PARAMS.wavelengths[i];
    }

    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      INT_PARAMS.sun_color.v[i] = sunRadianceAtWavelength(INT_PARAMS.wavelengths.v[i]);
      INT_PARAMS.rayleigh_scattering.v[i] = computeRayleighScattering(INT_PARAMS.wavelengths.v[i]);
      INT_PARAMS.ozone_absorption.v[i] = computeOzoneAbsorption(INT_PARAMS.wavelengths.v[i]);
    }

    printf("//    Wavelengths:          (");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%e", INT_PARAMS.wavelengths.v[i]);
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf(")\n");

    printf("//    Sun Color:           (");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%e", INT_PARAMS.sun_color.v[i]);
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf(")\n");

    printf("//    Rayleigh Scattering: (");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%e", INT_PARAMS.rayleigh_scattering.v[i]);
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf(")\n");

    printf("//    Ozone Absorption:    (");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%e", INT_PARAMS.ozone_absorption.v[i]);
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf(")\n");

    printf("//    Color Conversion Values:    \n//    {");
    printf("{");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%.7f", CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 1));
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf("},\n");
    printf("//     {");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%.7f", CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 2));
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf("},\n");
    printf("//     {");
    for (int i = 0; i < SKY_SPECTRUM_N; i++) {
      printf("%.7f", CieColorMatchingFunctionTableValue(INT_PARAMS.wavelengths.v[i], 3));
      if (i != SKY_SPECTRUM_N - 1) {
        printf(",");
      }
    }
    printf("}}\n");
#else
    INT_PARAMS.sun_color.r = sunRadianceAtWavelength(PARAMS.wavelength_red);
    INT_PARAMS.sun_color.g = sunRadianceAtWavelength(PARAMS.wavelength_green);
    INT_PARAMS.sun_color.b = sunRadianceAtWavelength(PARAMS.wavelength_blue);

    INT_PARAMS.rayleigh_scattering.r = computeRayleighScattering(PARAMS.wavelength_red);
    INT_PARAMS.rayleigh_scattering.g = computeRayleighScattering(PARAMS.wavelength_green);
    INT_PARAMS.rayleigh_scattering.b = computeRayleighScattering(PARAMS.wavelength_blue);

    printf("//    Rayleigh Scattering: (%e,%e,%e)\n", INT_PARAMS.rayleigh_scattering.r, INT_PARAMS.rayleigh_scattering.g, INT_PARAMS.rayleigh_scattering.b);

    INT_PARAMS.ozone_absorption.r = computeOzoneAbsorption(PARAMS.wavelength_red);
    INT_PARAMS.ozone_absorption.g = computeOzoneAbsorption(PARAMS.wavelength_green);
    INT_PARAMS.ozone_absorption.b = computeOzoneAbsorption(PARAMS.wavelength_blue);

    printf("//    Ozone Absorption: (%e,%e,%e)\n", INT_PARAMS.ozone_absorption.r, INT_PARAMS.ozone_absorption.g, INT_PARAMS.ozone_absorption.b);
#endif

    INT_PARAMS.tm_lut = (RGBF*)0;
    if (PARAMS.use_tm_lut) {
      computeTransmittance(&INT_PARAMS.tm_lut);
      printf("//    TransmittanceLUT Done.\n");
    }

    INT_PARAMS.ms_lut = (RGBF*)0;
    if (PARAMS.use_ms) {
      computeMultiScattering(&INT_PARAMS.ms_lut);
      printf("//    MultiScatteringLUT Done.\n");
    }
  }
  printf("//\n");

  if (resolution == SKY_MS_TEX_SIZE) {
    free(*outResult);
    *outResult = (float*) INT_PARAMS.ms_lut;
    return;
  }

  #pragma omp parallel for
  for (int x = 0; x < resolution; x++) {
    for (int y = 0; y < resolution; y++) {
      const vec3 ray = normalize_vector(pixelToDirection(x, y, resolution, view));

      sRGBF radiance;

      if (ray.x == 0.0f && ray.y == 0.0f && ray.z == 0.0f) {
        radiance.r = 0.0f;
        radiance.g = 0.0f;
        radiance.b = 0.0f;
      } else {
        radiance = sky_compute_atmosphere(pos, ray, FLT_MAX);
      }

      dst[x * resolution + y] = radiance;
    }
  }

  if (PARAMS.use_ms) {
    free(INT_PARAMS.ms_lut);
  }

  if (PARAMS.use_tm_lut) {
    free(INT_PARAMS.tm_lut);
  }
}