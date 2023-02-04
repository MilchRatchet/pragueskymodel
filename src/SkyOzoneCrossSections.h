#ifndef SKY_OZONE_CROSS_SECTIONS_H
#define SKY_OZONE_CROSS_SECTIONS_H

// Values from the Bruneton2017 implementation
// https://github.com/ebruneton/precomputed_atmospheric_scattering

// Number of molecules per cubic meter
static double ozoneCrossSectionMultipler = 300.0 * 2.687e20 / 15000.0;

// CrossSections in the range of 360nm to 930nm with 10nm steps
// Always round down when accessing
// Values are in 1 meter squared
static double ozoneCrossSection[38] = {
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26
};

// wavelength in 1 nm, output in 1 m^-1
static double ozoneCrossSectionSample(double wavelength) {
    if (wavelength > 749.0f) wavelength = 749.0f;
    if (wavelength < 361.0f) wavelength = 361.0f;

    const int index = (int)((wavelength - 360.0f) * 0.1f);

    return ozoneCrossSection[index] * ozoneCrossSectionMultipler;
}

#endif /* SKY_OZONE_CROSS_SECTIONS */