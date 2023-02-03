#include "SkyPathTracer.h"

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

#define SKY_RAYLEIGH_SCATTERING get_color(5.8f * 0.001f, 13.558f * 0.001f, 33.1f * 0.001f)
#define SKY_MIE_SCATTERING get_color(3.996f * 0.001f, 3.996f * 0.001f, 3.996f * 0.001f)
#define SKY_OZONE_SCATTERING 0.0f

#define SKY_RAYLEIGH_EXTINCTION SKY_RAYLEIGH_SCATTERING
#define SKY_MIE_EXTINCTION scale_color(SKY_MIE_SCATTERING, 1.11f)
#define SKY_OZONE_EXTINCTION get_color(0.65f * 0.001f, 1.881f * 0.001f, 0.085f * 0.001f)

#define SKY_RAYLEIGH_DISTRIBUTION 0.125f
#define SKY_MIE_DISTRIBUTION 0.833f

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

__device__ float henvey_greenstein(const float cos_angle, const float g) {
  return (1.0f - g * g) / (4.0f * PI * powf(1.0f + g * g - 2.0f * g * cos_angle, 1.5f));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Additional Structs Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

struct skyInternalParams {
  RGBF sun_color;
  vec3 sun_pos;
} typedef skyInternalParams;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//  Sky Section
///////////////////////////////////////////////////////////////////////////////////////////////////////

static skyPathTracerParams PARAMS;
static skyInternalParams INT_PARAMS;

__device__ float sky_rayleigh_phase(const float cos_angle) {
  return 3.0f * (1.0f + cos_angle * cos_angle) / (16.0f * 3.1415926535f);
}

__device__ float sky_density_falloff(const float height, const float density_falloff) {
  return expf(-height * density_falloff);
}

__device__ float sky_ozone_density(const float height) {
  if (!PARAMS.ozone_absorption)
    return 0.0f;

  if (height > 25.0f) {
    return fmaxf(0.0f, 1.0f - fabsf(height - 25.0f) * 0.04f);
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
  const float step_size = length / steps;
  RGBF density          = get_color(0.0f, 0.0f, 0.0f);
  float reach           = start;

  for (int i = 0; i < steps; i++) {
    const vec3 pos = add_vector(origin, scale_vector(ray, reach));

    const float height           = sky_height(pos);
    const float density_rayleigh = sky_density_falloff(height, SKY_RAYLEIGH_DISTRIBUTION);
    const float density_mie      = sky_density_falloff(height, SKY_MIE_DISTRIBUTION);
    const float density_ozone    = sky_ozone_density(height);

    RGBF D = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
    D      = add_color(D, scale_color(SKY_MIE_EXTINCTION, density_mie));
    D      = add_color(D, scale_color(SKY_OZONE_EXTINCTION, density_ozone));

    density = add_color(density, D);

    reach += step_size;
  }

  density = scale_color(density, -PARAMS.base_density * 0.5f * step_size);

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
    const float step_size = distance / steps;
    float reach           = start;

    for (int i = 0; i < steps; i++) {
      const vec3 pos = add_vector(origin, scale_vector(ray, reach));

      const float height = sky_height(pos);
      if (height < 0.0f || height > SKY_ATMO_HEIGHT) {
        reach += step_size;
        continue;
      }

      const float light_angle = sample_sphere_solid_angle(INT_PARAMS.sun_pos, SKY_SUN_RADIUS, pos);
      const vec3 ray_scatter  = normalize_vector(sub_vector(INT_PARAMS.sun_pos, pos));

      const float scatter_distance =
        (sph_ray_hit_p0(ray_scatter, pos, SKY_EARTH_RADIUS)) ? 0.0f : sph_ray_int_p0(ray_scatter, pos, SKY_ATMO_RADIUS);

      // If scatter_distance is 0.0 then all light is extinct
      // This is not very beautiful but it seems to be the easiest approach
      const RGBF extinction_sun = sky_extinction(pos, ray_scatter, 0.0f, scatter_distance);

      const float density_rayleigh = sky_density_falloff(height, SKY_RAYLEIGH_DISTRIBUTION);
      const float density_mie      = sky_density_falloff(height, SKY_MIE_DISTRIBUTION);
      const float density_ozone    = sky_ozone_density(height);

      const float cos_angle = fmaxf(0.0f, dot_product(ray, ray_scatter));

      const float phase_rayleigh = sky_rayleigh_phase(cos_angle);
      const float phase_mie      = henvey_greenstein(cos_angle, 0.8f);

      // Amount of light that reached pos
      RGBF S = scale_color(INT_PARAMS.sun_color, PARAMS.sun_strength);
      S      = mul_color(S, extinction_sun);

      // Amount of light that gets scattered towards camera at pos
      RGBF scattering = scale_color(SKY_RAYLEIGH_SCATTERING, density_rayleigh * phase_rayleigh);
      scattering      = add_color(scattering, scale_color(SKY_MIE_SCATTERING, density_mie * phase_mie));
      scattering      = scale_color(scattering, PARAMS.base_density * 0.5f * light_angle);

      S = mul_color(S, scattering);

      RGBF extinction = scale_color(SKY_RAYLEIGH_EXTINCTION, density_rayleigh);
      extinction      = add_color(extinction, scale_color(SKY_MIE_EXTINCTION, density_mie));
      extinction      = add_color(extinction, scale_color(SKY_OZONE_EXTINCTION, density_ozone));
      extinction      = scale_color(extinction, PARAMS.base_density * 0.5f);

      // Amount of light that gets lost along this step
      RGBF step_transmittance;
      step_transmittance.r = expf(-step_size * extinction.r);
      step_transmittance.g = expf(-step_size * extinction.g);
      step_transmittance.b = expf(-step_size * extinction.b);

      // Amount of light that gets scattered towards camera along this step
      S = mul_color(sub_color(S, mul_color(S, step_transmittance)), inv_color(extinction));

      result = add_color(result, mul_color(S, transmittance));

      transmittance = mul_color(transmittance, step_transmittance);

      reach += step_size;
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

void renderPathTracer(  const skyPathTracerParams        model,
                        const double                     albedo,
                        const double                     altitude,
                        const double                     azimuth,
                        const double                     elevation,
                        const int                        resolution,
                        const int                        view,
                        const double                     visibility,
                        float**                          outResult) {

  vec3 pos = {.x = 0.0f, .y = altitude + SKY_EARTH_RADIUS + 0.1f, .z = 0.0f};

  if (*outResult) {
    free(*outResult);
  }

  *outResult = malloc(sizeof(float) * resolution * resolution * 3);

  RGBF* dst = (RGBF*) *outResult;

  PARAMS = model;

  {
    INT_PARAMS.sun_color = get_color(1.0f, 1.0f, 1.0f);

    vec3 sun_pos = angles_to_direction(elevation, azimuth);
    sun_pos = normalize_vector(sun_pos);
    sun_pos = scale_vector(sun_pos, SKY_SUN_DISTANCE);
    sun_pos.y -= SKY_EARTH_RADIUS;

    INT_PARAMS.sun_pos = sun_pos;
  }

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

}