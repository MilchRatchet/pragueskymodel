#include "SkyPathTracer.h"

#include "SkyOzoneCrossSections.h"
#include "SkySpectrum2RGB.h"

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
#define SKY_ATMO_RADIUS SKY_ATMO_HEIGHT + SKY_EARTH_RADIUS

//#define SKY_RAYLEIGH_SCATTERING get_color(5.8f * 0.001f, 13.558f * 0.001f, 33.1f * 0.001f)
#define SKY_RAYLEIGH_SCATTERING INT_PARAMS.rayleigh_scattering
#define SKY_MIE_SCATTERING get_color(3.996f * 0.001f, 3.996f * 0.001f, 3.996f * 0.001f)
#define SKY_OZONE_SCATTERING 0.0f

#define SKY_RAYLEIGH_EXTINCTION SKY_RAYLEIGH_SCATTERING
#define SKY_MIE_EXTINCTION scale_color(SKY_MIE_SCATTERING, 1.11f)
//#define SKY_OZONE_EXTINCTION get_color(0.65f * 0.001f, 1.881f * 0.001f, 0.085f * 0.001f)
#define SKY_OZONE_EXTINCTION INT_PARAMS.ozone_absorption

#define SKY_RAYLEIGH_DISTRIBUTION (1.0f / 8.0f)
#define SKY_MIE_DISTRIBUTION (1.0f / 1.2f)

#define SKY_MS_TEX_SIZE 32

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Math Helper Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

struct float2 {
    float x;
    float y;
} typedef float2;

struct vec3 {
  float x;
  float y;
  float z;
} typedef vec3;

struct RGBF {
  float r;
  float g;
  float b;
} typedef RGBF;

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
} typedef skyInternalParams;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Sky Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

static skyPathTracerParams PARAMS;
static skyInternalParams INT_PARAMS;

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
  return expf(-height * (1.0f / PARAMS.rayleigh_height_falloff));
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

  return INSO + WASO;
}

__device__ float sky_ozone_density(const float height) {
  if (!PARAMS.ozone_absorption)
    return 0.0f;

  if (height > 25.0f) {
    return expf(-0.07f * (height - 25.0f))/*fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.04f)*/;
  }
  else {
    return fmaxf(0.1f, 1.0f - fabsf(height - 25.0f) * 0.066666667f);
  }
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

  density = scale_color(density, -PARAMS.base_density);

  return get_color(expf(density.r), expf(density.g), expf(density.b));
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
}

// Conversion of spectrum to sRGB as in Bruneton2017
// https://github.com/ebruneton/precomputed_atmospheric_scattering
__device__ RGBF sky_convert_wavelengths_to_sRGB(RGBF radiance) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float step_size = 1.0f;

  for (float lambda = SKY_WAVELENGTH_MIN; lambda < SKY_WAVELENGTH_MAX; lambda += step_size) {
    const float v = sky_interpolate_radiance_at_wavelength(radiance, lambda);
    x += CieColorMatchingFunctionTableValue(lambda, 1) * v;
    y += CieColorMatchingFunctionTableValue(lambda, 2) * v;
    z += CieColorMatchingFunctionTableValue(lambda, 3) * v;
  }

  const float MAX_LUMINOUS_EFFICACY = 683.0f / (SKY_WAVELENGTH_MAX - SKY_WAVELENGTH_MIN);

  RGBF result;
  result.r = MAX_LUMINOUS_EFFICACY * step_size * (3.2406f * x - 1.5372f * y - 0.4986f * z);
  result.g = MAX_LUMINOUS_EFFICACY * step_size * (-0.9689f * x + 1.8758f * y + 0.0415f * z);
  result.b = MAX_LUMINOUS_EFFICACY * step_size * (0.0557f * x - 0.2040f * y + 1.0570f * z);

  return result;
}

__device__ RGBF sky_compute_atmosphere(const vec3 origin, const vec3 ray, const float limit) {
  RGBF result = get_color(0.0f, 0.0f, 0.0f);

  float2 path = sky_compute_path(origin, ray, SKY_EARTH_RADIUS, SKY_ATMO_RADIUS);

  if (path.y == -FLT_MAX) {
    return result;
  }

  const float start    = path.x;
  const float distance = fminf(path.y, limit - start);

  RGBF transmittance = get_color(1.0f, 1.0f, 1.0f);

  if (distance > 0.0f) {
    const int steps       = PARAMS.steps;
    float step_size       = distance / steps;
    float reach           = start;

    const RGBF sun_radiance = scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength);

    for (float i = 0; i < steps; i += 1.0f) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size = newReach - reach;
      reach = newReach;

      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float height = sky_height(pos);

      const float light_angle = sample_sphere_solid_angle(INT_PARAMS.sun_pos, SKY_SUN_RADIUS, pos);
      const vec3 ray_scatter  = normalize_vector(sub_vector(INT_PARAMS.sun_pos, pos));

      const float shadow = sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;

      const float scatter_distance = sph_ray_int_p0(ray_scatter, pos, SKY_ATMO_RADIUS);

      const RGBF extinction_sun = sky_extinction(pos, ray_scatter, 0.0f, scatter_distance);

      const float density_rayleigh = sky_rayleigh_density(height) * PARAMS.density_rayleigh;
      const float density_mie      = sky_mie_density(height) * PARAMS.density_mie;
      const float density_ozone    = sky_ozone_density(height) * PARAMS.density_ozone;

      const float cos_angle = fmaxf(0.0f, dot_product(ray, ray_scatter));

      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = sky_mie_phase(cos_angle);

      const RGBF scattering_rayleigh = scale_color(SKY_RAYLEIGH_SCATTERING, density_rayleigh);
      const RGBF scattering_mie = scale_color(SKY_MIE_SCATTERING, density_mie);

      const RGBF extinction_rayleigh = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      const RGBF extinction_mie = scale_color(SKY_MIE_EXTINCTION, density_mie);
      const RGBF extinction_ozone = scale_color(SKY_OZONE_EXTINCTION, density_ozone);

      const RGBF scattering = add_color(scattering_rayleigh, scattering_mie);
      const RGBF extinction = add_color(add_color(extinction_rayleigh, extinction_mie), extinction_ozone);
      const RGBF phaseTimesScattering = add_color(scale_color(scattering_rayleigh, phase_rayleigh), scale_color(scattering_mie, phase_mie));

      const RGBF ssRadiance = scale_color(mul_color(extinction_sun, phaseTimesScattering), shadow);
      RGBF msRadiance = get_color(0.0f, 0.0f, 0.0f);

      if (PARAMS.use_ms) {
        const float sun_zenith_angle = dot_product(ray_scatter, normalize_vector(pos));

        const float ms_x = fmaxf(0.0f, fminf(1.0f, sun_zenith_angle * 0.5f + 0.5f));
        const float ms_y = fmaxf(0.0f, fminf(1.0f, height / SKY_ATMO_HEIGHT));

        int ms_ix = (int)(ms_x * (SKY_MS_TEX_SIZE - 1));
        int ms_iy = (int)(ms_y * (SKY_MS_TEX_SIZE - 1));

        int ms_ix1 = ms_ix + 1;
        int ms_iy1 = ms_iy + 1;

        RGBF ms00 = INT_PARAMS.ms_lut[ms_ix + ms_iy * SKY_MS_TEX_SIZE];
        RGBF ms01 = INT_PARAMS.ms_lut[ms_ix + ms_iy1 * SKY_MS_TEX_SIZE];
        RGBF ms10 = INT_PARAMS.ms_lut[ms_ix1 + ms_iy * SKY_MS_TEX_SIZE];
        RGBF ms11 = INT_PARAMS.ms_lut[ms_ix1 + ms_iy1 * SKY_MS_TEX_SIZE];

        RGBF msInterp = scale_color(ms00, (1.0f - ms_x) * (1.0f - ms_y));
        msInterp = add_color(msInterp, scale_color(ms01, (1.0f - ms_x) * ms_y));
        msInterp = add_color(msInterp, scale_color(ms10, ms_x * (1.0f - ms_y)));
        msInterp = add_color(msInterp, scale_color(ms11, ms_x * ms_y));

        msRadiance = mul_color(msInterp, scattering);
      }

      const RGBF S = mul_color(sun_radiance, add_color(ssRadiance, msRadiance));

      RGBF step_transmittance;
      step_transmittance.r = expf(-step_size * extinction.r);
      step_transmittance.g = expf(-step_size * extinction.g);
      step_transmittance.b = expf(-step_size * extinction.b);

      const RGBF Sint = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));

      result = add_color(result, mul_color(Sint, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);
    }
  }


  const float sun_hit   = sphere_ray_intersection(ray, origin, INT_PARAMS.sun_pos, SKY_SUN_RADIUS);
  const float earth_hit = sph_ray_int_p0(ray, origin, SKY_EARTH_RADIUS);

  if (earth_hit > sun_hit) {
    const vec3 sun_hit_pos  = add_vector(origin, scale_vector(ray, sun_hit));
    const float limb_factor = 1.0f + dot_product(normalize_vector(sub_vector(sun_hit_pos, INT_PARAMS.sun_pos)), ray);
    const float mu          = sqrtf(1.0f - limb_factor * limb_factor);

    const RGBF limb_color = get_color(0.397f, 0.503f, 0.652f);

    const RGBF limb_darkening = get_color(powf(mu, limb_color.r), powf(mu, limb_color.g), powf(mu, limb_color.b));

    RGBF S = mul_color(transmittance, scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength));
    S      = mul_color(S, limb_darkening);

    result = add_color(result, S);
  }

  return result;
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

static msScatteringResult computeMultiScatteringIntegration(vec3 origin, vec3 ray, vec3 sun_dir) {
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
    float reach           = start + PARAMS.sampling_offset * step_size;

    const float cos_angle = dot_product(ray, sun_dir);
    const float phase_uniform = 1.0f / (4.0f * PI);

    RGBF transmittance = get_color(1.0f, 1.0f, 1.0f);

    for (float i = 0; i < steps; i += 1.0f) {
      const float newReach = start + distance * (i + 0.3f) / steps;
      step_size = newReach - reach;
      reach = newReach;

      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float scatter_distance = sph_ray_int_p0(sun_dir, pos, SKY_ATMO_RADIUS);
      const RGBF extinction_sun = sky_extinction(pos, sun_dir, 0.0f, scatter_distance);

      const float height = sky_height(pos);
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
      const RGBF phaseTimesScattering = scale_color(scattering, phase_uniform);

      const float shadow = sph_ray_hit_p0(sun_dir, pos, SKY_EARTH_RADIUS) ? 0.0f : 1.0f;
      const RGBF S = scale_color(mul_color(extinction_sun, phaseTimesScattering), shadow);

      RGBF step_transmittance;
      step_transmittance.r = expf(-step_size * extinction.r);
      step_transmittance.g = expf(-step_size * extinction.g);
      step_transmittance.b = expf(-step_size * extinction.b);

      const RGBF ssInt = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));
      const RGBF msInt = mul_color(sub_color(scattering, mul_color(scattering, step_transmittance)), inv_color(extinction));

      result.L = add_color(result.L, mul_color(ssInt, transmittance));
      result.multiScatterAs1 = add_color(result.multiScatterAs1, mul_color(msInt, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);
    }
  }

  return result;
}

#define saturatef(x) fmaxf(0.0f, fminf(1.0f, (x)))

// Hillaire 2020
static void computeMultiScattering(RGBF** msTex) {
  if (*msTex) {
    free(*msTex);
  }

  *msTex = malloc(sizeof(RGBF) * SKY_MS_TEX_SIZE * SKY_MS_TEX_SIZE);

  #pragma omp parallel for
  for (int x = 0; x < SKY_MS_TEX_SIZE; x++) {
    for (int y = 0; y < SKY_MS_TEX_SIZE; y++) {
      const float fx = ((float)x + 0.5f) / SKY_MS_TEX_SIZE;
      const float fy = ((float)y + 0.5f) / SKY_MS_TEX_SIZE;

      float cos_angle = fx * 2.0f - 1.0f;
      vec3 sun_dir = get_vector(0.0f, cos_angle, -sinf(acosf(cos_angle)));
      float height = SKY_EARTH_RADIUS + saturatef(fy) * SKY_ATMO_HEIGHT + 0.001f;

      vec3 pos = get_vector(0.0f, height, 0.0f);

      RGBF inscatteredLuminance = get_color(0.0f, 0.0f, 0.0f);
      RGBF multiScattering = get_color(0.0f, 0.0f, 0.0f);

      const float sqrt_sample = 8.0f;

      for (int i = 0; i < 64; i++) {
        float a = 0.5f + i / 8;
        float b = 0.5f + (i - ((i/8) * 8));
        float randA = a / sqrt_sample;
        float randB = b / sqrt_sample;
        float theta = 2.0f * PI * randA;
        float phi = acosf(1.0f - 2.0f * randB);
        vec3 ray = angles_to_direction(phi, theta);

        msScatteringResult result = computeMultiScatteringIntegration(pos, ray, sun_dir);

        inscatteredLuminance = add_color(inscatteredLuminance, result.L);
        multiScattering = add_color(multiScattering, result.multiScatterAs1);
      }

      inscatteredLuminance = scale_color(inscatteredLuminance, 1.0f / (sqrt_sample * sqrt_sample));
      multiScattering = scale_color(multiScattering, 1.0f / (sqrt_sample * sqrt_sample));

      RGBF multiScatteringContribution = inv_color(sub_color(get_color(1.0f, 1.0f, 1.0f), multiScattering));

      RGBF L = mul_color(inscatteredLuminance, multiScatteringContribution);

      INT_PARAMS.ms_lut[x + y * SKY_MS_TEX_SIZE] = L;
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

  vec3 pos = {.x = 0.0f, .y = altitude + SKY_EARTH_RADIUS + 0.001f, .z = 0.0f};

  if (*outResult) {
    free(*outResult);
  }

  *outResult = malloc(sizeof(float) * resolution * resolution * 3);

  RGBF* dst = (RGBF*) *outResult;

  PARAMS = model;

  printf("//////////////////////////////////////\n");
  printf("//\n");
  printf("//  Path Tracer Data:\n");
  printf("//\n");
  {
    INT_PARAMS.sun_color = get_color(1.0f, 1.0f, 1.0f);//get_color(1.474f, 1.8504f, 1.91198f);

    vec3 sun_pos = angles_to_direction(elevation, azimuth);
    sun_pos = normalize_vector(sun_pos);
    sun_pos = scale_vector(sun_pos, SKY_SUN_DISTANCE);
    sun_pos.y -= SKY_EARTH_RADIUS;

    INT_PARAMS.sun_pos = sun_pos;

    INT_PARAMS.rayleigh_scattering.r = computeRayleighScattering(PARAMS.wavelength_red);
    INT_PARAMS.rayleigh_scattering.g = computeRayleighScattering(PARAMS.wavelength_green);
    INT_PARAMS.rayleigh_scattering.b = computeRayleighScattering(PARAMS.wavelength_blue);

    printf("//    Rayleigh Scattering: (%e,%e,%e)\n", INT_PARAMS.rayleigh_scattering.r, INT_PARAMS.rayleigh_scattering.g, INT_PARAMS.rayleigh_scattering.b);

    INT_PARAMS.ozone_absorption.r = computeOzoneAbsorption(PARAMS.wavelength_red);
    INT_PARAMS.ozone_absorption.g = computeOzoneAbsorption(PARAMS.wavelength_green);
    INT_PARAMS.ozone_absorption.b = computeOzoneAbsorption(PARAMS.wavelength_blue);

    printf("//    Ozone Absorption: (%e,%e,%e)\n", INT_PARAMS.ozone_absorption.r, INT_PARAMS.ozone_absorption.g, INT_PARAMS.ozone_absorption.b);

    INT_PARAMS.ms_lut = (RGBF*)0;
    if (PARAMS.use_ms) {
      computeMultiScattering(&INT_PARAMS.ms_lut);
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

      RGBF radiance;

      if (ray.x == 0.0f && ray.y == 0.0f && ray.z == 0.0f) {
        radiance = get_color(0.0f, 0.0f, 0.0f);
      } else {
        radiance = sky_compute_atmosphere(pos, ray, FLT_MAX);
      }

      dst[x * resolution + y] = radiance;
    }
  }

  if (PARAMS.use_ms) {
    free(INT_PARAMS.ms_lut);
  }
}