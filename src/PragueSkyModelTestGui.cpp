// Copyright 2022 Charles University
// SPDX-License-Identifier: Apache-2.0

// For saving the result to an EXR file.
#pragma warning(push)
#pragma warning(disable : 26495)
#pragma warning(disable : 26451)
#pragma warning(disable : 6386)
#pragma warning(disable : 6387)
#pragma warning(disable : 6385)
#pragma warning(disable : 6011)
#pragma warning(disable : 6001)
#pragma warning(disable : 4018)
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>
#pragma warning(pop)

// For using the sky model.
#include "PragueSkyModelTest.hpp"
#include "SkyPathTracer.h"

// For GUI.
#include <chrono>
#include <sstream>
#include "imgui.h"
#include "imfilebrowser.h"
#ifdef _WIN32
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#else
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl2.h"
#include <stdio.h>
#include <SDL.h>
#include <SDL_opengl.h>
#endif


/////////////////////////////////////////////////////////////////////////////////////
// GUI helper functions
/////////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
static ID3D11Device*           g_pd3dDevice           = NULL;
static ID3D11DeviceContext*    g_pd3dDeviceContext    = NULL;
static IDXGISwapChain*         g_pSwapChain           = NULL;
static ID3D11RenderTargetView* g_mainRenderTargetView = NULL;

void createRenderTarget() {
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void cleanupRenderTarget() {
    if (g_mainRenderTargetView) {
        g_mainRenderTargetView->Release();
        g_mainRenderTargetView = NULL;
    }
}

bool createDeviceD3D(HWND hWnd) {
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount                        = 2;
    sd.BufferDesc.Width                   = 0;
    sd.BufferDesc.Height                  = 0;
    sd.BufferDesc.Format                  = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator   = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags                              = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage                        = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow                       = hWnd;
    sd.SampleDesc.Count                   = 1;
    sd.SampleDesc.Quality                 = 0;
    sd.Windowed                           = TRUE;
    sd.SwapEffect                         = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    // createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
    D3D_FEATURE_LEVEL       featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };
    if (D3D11CreateDeviceAndSwapChain(NULL,
                                      D3D_DRIVER_TYPE_HARDWARE,
                                      NULL,
                                      createDeviceFlags,
                                      featureLevelArray,
                                      2,
                                      D3D11_SDK_VERSION,
                                      &sd,
                                      &g_pSwapChain,
                                      &g_pd3dDevice,
                                      &featureLevel,
                                      &g_pd3dDeviceContext) != S_OK)
        return false;

    createRenderTarget();

    return true;
}

void cleanupDeviceD3D() {
    cleanupRenderTarget();
    if (g_pSwapChain) {
        g_pSwapChain->Release();
        g_pSwapChain = NULL;
    }
    if (g_pd3dDeviceContext) {
        g_pd3dDeviceContext->Release();
        g_pd3dDeviceContext = NULL;
    }
    if (g_pd3dDevice) {
        g_pd3dDevice->Release();
        g_pd3dDevice = NULL;
    }
}

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND   hWnd,
                                                             UINT   msg,
                                                             WPARAM wParam,
                                                             LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg) {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED) {
            cleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0,
                                        (UINT)LOWORD(lParam),
                                        (UINT)HIWORD(lParam),
                                        DXGI_FORMAT_UNKNOWN,
                                        0);
            createRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
#endif

static void tonemapACES(float& r, float& g, float& b) {
    float cr = 0.59719f * r + 0.35458f * g + 0.04823f * b;
    float cg = 0.07600f * r + 0.90834f * g + 0.01566f * b;
    float cb = 0.02840f * r + 0.13383f * g + 0.83777f * b;

    float ar = cr + 0.0245786f;
    float ag = cg + 0.0245786f;
    float ab = cb + 0.0245786f;
    ar = ar * cr;
    ag = ag * cg;
    ab = ab * cb;
    ar = ar - 0.000090537f;
    ag = ag - 0.000090537f;
    ab = ab - 0.000090537f;

    float br = cr * 0.983729f;
    float bg = cg * 0.983729f;
    float bb = cb * 0.983729f;
    br = br + 0.432951f;
    bg = bg + 0.432951f;
    bb = bb + 0.432951f;
    br = br * cr;
    bg = bg * cg;
    bb = bb * cb;
    br = br + 0.238081f;
    bg = bg + 0.238081f;
    bb = bb + 0.238081f;
    br = 1.0f / br;
    bg = 1.0f / bg;
    bb = 1.0f / bb;

    cr = ar * br;
    cg = ag * bg;
    cb = ab * bb;

    r = 1.60475f * cr - 0.53108f * cg - 0.07367f * cb;
    g = -0.10208f * cr + 1.10813f * cg - 0.00605f * cb;
    b = -0.00327f * cr - 0.07276f * cg + 1.07602f * cb;
}

static void tonemapReinhard(float& r, float& g, float& b) {
    const float f = 1.0f / (1.0f + 0.2126f * r + 0.7152f * g + 0.0722f * b);
    r *= f;
    g *= f;
    b *= f;
}

static void tonemapUncharted2_helper(float& r, float& g, float& b) {
    const float a = 0.15f;
    const float b2 = 0.50f;
    const float c = 0.10f;
    const float d = 0.20f;
    const float e = 0.02f;
    const float f = 0.30f;

    const float pr = ((r * (a * r + c * b2) + d * e) / (r * (a * r + b2) + d * f)) - e / f;
    const float pg = ((g * (a * g + c * b2) + d * e) / (g * (a * g + b2) + d * f)) - e / f;
    const float pb = ((b * (a * b + c * b2) + d * e) / (b * (a * b + b2) + d * f)) - e / f;

    r = pr;
    g = pg;
    b = pb;
}

static void tonemapUncharted2(float& r, float& g, float& b) {
    r *= 2.0f;
    g *= 2.0f;
    b *= 2.0f;

    tonemapUncharted2_helper(r, g, b);

    float sr = 11.2f;
    float sg = 11.2f;
    float sb = 11.2f;
    tonemapUncharted2_helper(sr, sg, sb);

    sr = 1.0f / sr;
    sg = 1.0f / sg;
    sb = 1.0f / sb;

    r *= sr;
    g *= sg;
    b *= sb;
}

static float linearRGB_to_sRGB(const float value) {
    /*
     * Original: std:pow(value, 1.0f / 2.2f);
     * This is what I use in Luminary
     */
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    else {
        return 1.055f * std::pow(value, 0.416666666667f) - 0.055f;
    }
}

// src is 3 values but dst is 4, don't ask, I don't make the rules
static void colorFloatTo8Bit(const float* src, uint8_t* dst, const float exposure, void (*tonemapFunc)(float&,float&,float&)) {
    float r = src[0] * exposure;
    float g = src[1] * exposure;
    float b = src[2] * exposure;

    if (tonemapFunc) {
        tonemapFunc(r,g,b);
    }

    r = linearRGB_to_sRGB(r);
    g = linearRGB_to_sRGB(g);
    b = linearRGB_to_sRGB(b);

    dst[0] = (uint8_t)(std::floor(std::clamp(r * 255.9f, 0.0f, 255.9f)));
    dst[1] = (uint8_t)(std::floor(std::clamp(g * 255.9f, 0.0f, 255.9f)));
    dst[2] = (uint8_t)(std::floor(std::clamp(b * 255.9f, 0.0f, 255.9f)));
    dst[3] = 255;
}

void convertToTexture(const float*              image,
                      const int                 resolution,
                      const float               exposure,
                      const int                 tonemapChoice,
                      void**                    texture) {
    // Apply exposure and convert float RGB to byte RGBA
    void (*tonemapFunc)(float&,float&,float&);
    switch (tonemapChoice) {
        case 1:
            tonemapFunc = tonemapACES;
            break;
        case 2:
            tonemapFunc = tonemapReinhard;
            break;
        case 3:
            tonemapFunc = tonemapUncharted2;
            break;
        default:
            tonemapFunc = 0;
            break;
    }
    std::vector<uint8_t> rawData;
    rawData.resize(size_t(resolution) * resolution * 4);
    const float expMult = std::pow(2.f, exposure);
    for (int x = 0; x < resolution; x++) {
        for (int y = 0; y < resolution; y++) {
            colorFloatTo8Bit(&image[(size_t(x) * resolution + y) * 3], &rawData[(size_t(x) * resolution + y) * 4], expMult, tonemapFunc);
        }
    }

#ifdef _WIN32
    // Create texture
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width            = resolution;
    desc.Height           = resolution;
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_DEFAULT;
    desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags   = 0;

    ID3D11Texture2D*       pTexture = NULL;
    D3D11_SUBRESOURCE_DATA subResource;
    subResource.pSysMem          = rawData.data();
    subResource.SysMemPitch      = desc.Width * 4;
    subResource.SysMemSlicePitch = 0;
    g_pd3dDevice->CreateTexture2D(&desc, &subResource, &pTexture);

    // Create texture view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format                    = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension             = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels       = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    g_pd3dDevice->CreateShaderResourceView(pTexture, &srvDesc, (ID3D11ShaderResourceView**)texture);
    pTexture->Release();
#else
    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,
                    GL_TEXTURE_WRAP_S,
                    GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 resolution,
                 resolution,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 rawData.data());

    *texture = (void*)(intptr_t)image_texture;
#endif
}

void helpMarker(const char* desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void errorMarker(const char* desc) {
    ImGui::Text("ERROR!");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// Main
/////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    PragueSkyModel                  skyModel;
    std::vector<std::vector<float>> result;
    float*                          resultPt = NULL;
    void*                           texture = NULL;
    const char* modes[] = { "Sky radiance", "Sun radiance", "Polarisation", "Transmittance" };
    const char* tonemaps[] = { "None", "ACES", "Reinhard", "Uncharted2" };
    const char* views[] = { "Up-facing fisheye", "Side-facing fisheye" };
    char        label[150];
    const char*        visibilitiesToLoad[] = { "Everything",
                                         "only visibilities 20.0 - 27.6 km",
                                         "only visibilities 27.6 - 40.0 km",
                                         "only visibilities 40.0 - 59.4 km",
                                         "only visibilities 59.4 - 90.0 km",
                                         "only visibilities 90.0 - 131.8 km" };

    // Default ranges
    PragueSkyModel::AvailableData available;
    available.albedoMin     = 0.0;
    available.albedoMax     = 1.0;
    available.altitudeMin   = 0.0;
    available.altitudeMax   = 15000.0;
    available.elevationMin  = -4.2;
    available.elevationMax  = 90.0;
    available.visibilityMin = 20.0;
    available.visibilityMax = 131.8;
    available.polarisation  = true;
    available.channels      = SPECTRUM_CHANNELS;
    available.channelStart  = SPECTRUM_WAVELENGTHS[0] - 0.5 * SPECTRUM_STEP;
    available.channelWidth  = SPECTRUM_STEP;

    // The full window and the input subwindow dimensions.
#ifdef _WIN32
    const int         windowWidthInput  = 440;
    const int         windowHeightInput = 1080;
    const std::string dirSeparator      = "\\";
    const int         windowWidthImage  = 1080;
    const int         windowHeightImage = 1080;
#else
    const int         windowWidthInput  = 440;
    const int         windowHeightInput = 800;
    const std::string dirSeparator      = "/";
    const int         windowWidthImage  = 1080;
    const int         windowHeightImage = 1080;
#endif

    const int windowWidthFull  = windowWidthInput + windowWidthImage;
    const int windowHeightFull = std::max(windowHeightInput, windowHeightImage);

#ifdef _WIN32
    // Create application window
    WNDCLASSEX wc = { sizeof(WNDCLASSEX),     CS_CLASSDC, WndProc, 0L,   0L,
                      GetModuleHandle(NULL),  NULL,       NULL,    NULL, NULL,
                      _T("Prague Sky Model"), NULL };
    ::RegisterClassEx(&wc);
    HWND hwnd = ::CreateWindow(wc.lpszClassName,
                               _T("Prague Sky Model"),
                               WS_OVERLAPPEDWINDOW,
                               100,
                               100,
                               windowWidthFull,
                               windowHeightFull,
                               NULL,
                               NULL,
                               wc.hInstance,
                               NULL);

    // Initialize Direct3D
    if (!createDeviceD3D(hwnd)) {
        cleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);
#else
    // Setup window
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_WindowFlags window_flags =
        (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Window*   window     = SDL_CreateWindow("Prague Sky Model",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          windowWidthFull,
                                          windowHeightFull,
                                          window_flags);
    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync
#endif

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
#ifdef _WIN32
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);
#else
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL2_Init();
#endif

    // Flags for a window with user input
    ImGuiWindowFlags inputWindowFlags = 0;
    inputWindowFlags |= ImGuiWindowFlags_NoTitleBar;
    inputWindowFlags |= ImGuiWindowFlags_NoScrollbar;
    inputWindowFlags |= ImGuiWindowFlags_NoMove;
    inputWindowFlags |= ImGuiWindowFlags_NoResize;
    inputWindowFlags |= ImGuiWindowFlags_NoCollapse;

    // Flags for a window displaying output
    ImGuiWindowFlags outputWindowFlags = 0;
    outputWindowFlags |= ImGuiWindowFlags_NoTitleBar;
    outputWindowFlags |= ImGuiWindowFlags_NoMove;
    outputWindowFlags |= ImGuiWindowFlags_NoResize;
    outputWindowFlags |= ImGuiWindowFlags_NoCollapse;
    outputWindowFlags |= ImGuiWindowFlags_NoBackground;
    outputWindowFlags |= ImGuiWindowFlags_HorizontalScrollbar;

    // Create a file browser instance for selecting dataset file
    ImGui::FileBrowser fileDialogOpen;
    fileDialogOpen.SetTitle("Select dataset file");
    fileDialogOpen.SetTypeFilters({ ".dat" });

    // Create a file browser instance for selecting output file
    ImGui::FileBrowser fileDialogSave(ImGuiFileBrowserFlags_EnterNewFilename |
                                      ImGuiFileBrowserFlags_CreateNewDir);
    fileDialogSave.SetTitle("Select output file");

    // Main loop
    bool done = false;
    while (!done) {
#ifdef _WIN32
        // Poll and handle messages (inputs, window resize, etc.)
        MSG msg;
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                done = true;
            }
        }
        if (done) {
            break;
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
#else
        // Poll and handle events (inputs, window resize, etc.)
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window)) {
                done = true;
            }
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
#endif
        ImGui::NewFrame();

        // All values modified by the GUI
        static float       albedo             = 0.0f;
        static float       altitude           = 0.0f;
        static bool        autorender         = false;
        static float       azimuth            = 0.5f * 3.14159265359f;
        static int         channelMode        = 0;
        static std::string datasetName        = "PragueSkyModelDatasetGround.dat";
        static std::string datasetPath        = "PragueSkyModelDatasetGround.dat";
        static bool        displayError       = false;
        static bool        displayPathTrace   = false;
        static float       elevation          = 0.0f;
        static float       exposure           = 0.f;
        static bool        loaded             = false;
        static bool        loading            = false;
        static std::string loadError          = "";
        static std::string outputName         = "test.exr";
        static std::string outputPath         = "test.exr";
        static int         resolution         = 500;
        static int         renderedResolution = resolution;
        static bool        rendered           = false;
        static bool        rendering          = false;
        static std::string renderError        = "";
        static long long   renderTime         = 0;
        static long long   renderTimePt       = 0;
        static bool        saved              = false;
        static std::string saveError          = "";
        static int         tonemapper         = 0;
        static bool        updateTexture      = false;
        static int         view               = 1;
        static float       visibility         = 59.4f;
        static int         visibilityToLoad   = 0;
        static int         wavelength         = 280;
        static float       zoom               = 2.0f;
        static skyPathTracerParams ptParams   = {
                1, /* OZONE_ABSORPTION */
                40, /* SHADOW_STEPS */
                1.0f, /* BASE_DENSITY */
                10.0f, /* SUN_STRENGTH */
                40, /* STEPS */
                0, /* USE CS MIE */
                0.8f, /* MIE G */
                1.0f, /* Rayleigh Density*/
                1.0f, /* Mie Density*/
                1.0f, /* Ozone Density*/
                8.0f, /* Rayleigh Falloff */
                1.7f, /* Mie Falloff */
                635.0f, /* Wavelength Red */
                550.0f, /* Wavelength Green */
                415.0f, /* Wavelength Blue */
                0.03f, /* Carbondioxide Percent */
                1, /* Use Multiscattering */
                59.4f, /* Ground Visibility */
                1.0f, /* Multiscattering Factor */
                0, /* Convert Spectrum */
                1, /* Use Transmittance LUT */
                1, /* Ground */
                0.3f, /* Ground Albedo */
                1, /* Static Sun Solid Angle */
                15.0f /* Ozone Layer Thickness */
        };

        // Input window
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(windowWidthInput, windowHeightInput));
            ImGui::Begin("Prague Sky Model", NULL, inputWindowFlags);

            /////////////////////////////////////////////
            // Dataset section
            /////////////////////////////////////////////

            // Dataset section begin
            ImGui::Text("Dataset:");

            // Dataset file selection
            if (ImGui::Button(datasetName.c_str(), ImVec2(ImGui::CalcItemWidth(), 20))) {
                fileDialogOpen.Open();
            }
            ImGui::SameLine();
            ImGui::Text("file");
            ImGui::SameLine();
            helpMarker("Sky model dataset file");
            fileDialogOpen.Display();
            if (fileDialogOpen.HasSelected()) {
                datasetPath = fileDialogOpen.GetSelected().string();
                datasetName = datasetPath.substr(datasetPath.find_last_of(dirSeparator) + 1);
                fileDialogOpen.ClearSelected();
            }

            // Selection of visibilities to load
            ImGui::Combo("part to load", &visibilityToLoad, visibilitiesToLoad, IM_ARRAYSIZE(visibilitiesToLoad));
            ImGui::SameLine();
            helpMarker("What portion of the dataset should be loaded");

            // Load button
            if (loading) {
                try {
                    double singleVisibility = 0.0;
                    switch (visibilityToLoad) {
                    case 1: // 20.0 - 27.6
                        singleVisibility = 23.8;
                        break;
                    case 2: // 27.6 - 40.0
                        singleVisibility = 33.8;
                        break;
                    case 3: // 40.0 - 59.4
                        singleVisibility = 49.7;
                        break;
                    case 4: // 59.4 - 90.0
                        singleVisibility = 74.7;
                        break;
                    case 5: // 90.0 - 131.8
                        singleVisibility = 110.9;
                        break;
                    default:
                        singleVisibility = 0.0;
                        break;
                    }
                    skyModel.initialize(datasetPath, singleVisibility);
                    loaded    = true;
                    available = skyModel.getAvailableData();
                } catch (std::exception& e) {
                    loadError = e.what();
                    loaded    = false;
                }
                loading = false;
            }
            if (ImGui::Button("Load")) {
                loading = true;
                ImGui::SameLine();
                ImGui::Text("Loading ...");
            }
            if (loaded && !loading) {
                ImGui::SameLine();
                ImGui::Text("OK");
            } else if (!loadError.empty() && !loading) {
                ImGui::SameLine();
                errorMarker(loadError.c_str());
            }

            // Dataset section end
            ImGui::Dummy(ImVec2(0.0f, 1.0f));
            ImGui::Separator();

            /////////////////////////////////////////////
            // Configuration section
            /////////////////////////////////////////////

            // Configuration section begin
            if (!loaded || loading) {
                ImGui::BeginDisabled(true);
            }
            ImGui::Text("Configuration:");

            // Parameters
            if (ImGui::SliderFloat("albedo",
                                   &albedo,
                                   available.albedoMin,
                                   available.albedoMax,
                                   "%.2f",
                                   ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            sprintf(label,
                    "Ground albedo, value in range [%.1f, %.1f]",
                    available.albedoMin,
                    available.albedoMax);
            helpMarker(label);
            if (ImGui::SliderFloat("altitude",
                               &altitude,
                               available.altitudeMin,
                               available.altitudeMax,
                               "%.0f m",
                                   ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            sprintf(label,
                    "Altitude of view point in meters, value in range [%.1f, %.1f]",
                    available.altitudeMin,
                    available.altitudeMax);
            helpMarker(label);
            if (ImGui::SliderAngle("azimuth",
                                   &azimuth,
                                   0.0f,
                                   360.0f,
                                   "%.1f deg",
                                   ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            helpMarker("Sun azimuth at view point in degrees, value in range [0, 360]");
            if (ImGui::SliderAngle("elevation",
                               &elevation,
                               available.elevationMin,
                               available.elevationMax,
                               "%.1f deg",
                                   ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            sprintf(label,
                    "Sun elevation at view point in degrees, value in range [%.1f, %.1f]",
                    available.elevationMin,
                    available.elevationMax);
            helpMarker(label);
            if (ImGui::DragInt("resolution",
                               &resolution,
                               1,
                               1,
                               10000,
                               "%d px",
                               ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            helpMarker("Length of resulting square image size in pixels, value in range [1, 10000]");
            if (ImGui::SliderFloat("visibility",
                               &visibility,
                               available.visibilityMin,
                               available.visibilityMax,
                               "%.1f km",
                                   ImGuiSliderFlags_AlwaysClamp) &&
                autorender)
                rendering = true;
            ImGui::SameLine();
            sprintf(label,
                    "Horizontal visibility (meteorological range) at ground level in kilometers, value in "
                    "range [%.1f, %.1f]",
                    available.visibilityMin,
                    available.visibilityMax);
            helpMarker(label);
            if (ImGui::Combo("view", &view, views, IM_ARRAYSIZE(views)) && autorender)
                rendering = true;
            ImGui::SameLine();
            helpMarker("Rendered view");

            // Render button
            if (rendering) {
                try {
                    // Render and measure how long it takes
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    render(skyModel,
                           albedo,
                           altitude,
                           azimuth,
                           elevation,
                           resolution,
                           View(view),
                           visibility,
                           result);
                    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                    renderTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                    begin = std::chrono::steady_clock::now();
                    renderPathTracer(ptParams,
                           albedo,
                           altitude,
                           azimuth,
                           elevation,
                           resolution,
                           view,
                           visibility,
                           &resultPt);
                    end = std::chrono::steady_clock::now();
                    renderTimePt = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                    rendered   = true;
                    updateTexture      = true;
                    renderedResolution = resolution;
                } catch (std::exception& e) {
                    renderError = e.what();
                    rendered    = false;
                }
                rendering = false;
            }
            if (ImGui::Button("Render")) {
                rendering = true;
                ImGui::SameLine();
                ImGui::Text("Rendering ...");
            }
            ImGui::SameLine();
            ImGui::Checkbox("Auto-update", &autorender);
            if (rendered && !rendering) {
                ImGui::SameLine();
                std::ostringstream out;
                out << "OK (" << renderTime << "|" << renderTimePt << " ms)";
                ImGui::Text(out.str().c_str());
            } else if (!renderError.empty() && !rendering) {
                ImGui::SameLine();
                errorMarker(renderError.c_str());
            }

            // Configuration section end
            ImGui::Dummy(ImVec2(0.0f, 1.0f));
            ImGui::Separator();
            if (!loaded || loading) {
                ImGui::EndDisabled();
            }

            /////////////////////////////////////////////
            // Path Tracer Configuration section
            /////////////////////////////////////////////

            if (!loaded || loading) {
                ImGui::BeginDisabled(true);
            }
            ImGui::Text("Path Tracer Configuration:");
            ImGui::SliderFloat("density", &ptParams.base_density, 0.0f, 10.0f, "%.1f");
            ImGui::SameLine();
            helpMarker("Density of the atmosphere");
            ImGui::SliderFloat("sun intensity", &ptParams.sun_strength, 0.0f, 500.0f, "%.1f");
            ImGui::SameLine();
            helpMarker("Intensity of sun");
            ImGui::Checkbox("ozone absorption", (bool*)&ptParams.ozone_absorption);
            ImGui::DragInt("steps",
                               &ptParams.steps,
                               1,
                               1,
                               10000,
                               "%d",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::DragInt("shadow steps",
                               &ptParams.shadow_steps,
                               1,
                               1,
                               10000,
                               "%d",
                               ImGuiSliderFlags_AlwaysClamp);
            ImGui::Checkbox("use cs mie phase", (bool*)&ptParams.use_cs_mie);
            ImGui::SliderFloat("mie g", &ptParams.mie_g, -1.0f, 1.0f, "%.3f");
            ImGui::SliderFloat("density rayleigh", &ptParams.density_rayleigh, 0.0f, 10.0f, "%.3f");
            ImGui::SliderFloat("density mie", &ptParams.density_mie, 0.0f, 10.0f, "%.3f");
            ImGui::SliderFloat("density ozone", &ptParams.density_ozone, 0.0f, 10.0f, "%.3f");
            ImGui::SliderFloat("height falloff rayleigh", &ptParams.rayleigh_height_falloff, 0.1f, 20.0f, "%.3f");
            ImGui::SliderFloat("height falloff mie", &ptParams.mie_height_falloff, 0.1f, 20.0f, "%.3f");
            ImGui::SliderFloat("wavelength red", &ptParams.wavelength_red, 380.0f, 750.0f, "%.1f");
            ImGui::SliderFloat("wavelength green", &ptParams.wavelength_green, 380.0f, 750.0f, "%.1f");
            ImGui::SliderFloat("wavelength blue", &ptParams.wavelength_blue, 380.0f, 750.0f, "%.1f");
            ImGui::SliderFloat("CO2 percentage", &ptParams.carbondioxide_percent, 0.0f, 1.0f, "%.3f");
            ImGui::Checkbox("use multiscattering", (bool*)&ptParams.use_ms);
            ImGui::SliderFloat("ground visibility", &ptParams.ground_visibility, 20.0f, 131.8f, "%.1f km");
            ImGui::SliderFloat("multiscattering factor", &ptParams.ms_factor, 0.0f, 2.0f, "%.3f");
            ImGui::Checkbox("convert spectrum to rgb", (bool*)&ptParams.convertSpectrum);
            ImGui::Checkbox("use transmittance lut", (bool*)&ptParams.use_tm_lut);
            ImGui::Checkbox("ground", (bool*)&ptParams.ground);
            if (!ptParams.ground) {
                ImGui::BeginDisabled();
            }
            ImGui::SliderFloat("ground albedo", &ptParams.ground_albedo, 0.0f, 1.0f, "%.3f");
            if (!ptParams.ground) {
                ImGui::EndDisabled();
            }
            ImGui::Checkbox("use static sun solid angle", (bool*)&ptParams.use_static_sun_solid_angle);
            ImGui::SliderFloat("ozone layer thickness", &ptParams.ozone_layer_thickness, 0.1f, 25.0f, "%.1f km");

            ImGui::Dummy(ImVec2(0.0f, 1.0f));
            ImGui::Separator();
            if (!loaded || loading) {
                ImGui::EndDisabled();
            }

            /////////////////////////////////////////////
            // Display section
            /////////////////////////////////////////////

            // Display section begin
            if (!rendered || rendering) {
                ImGui::BeginDisabled(true);
            }
            ImGui::Text("Display:");

            // Parameters
            if (ImGui::BeginCombo("tonemap", tonemaps[tonemapper])) {
                for (int i = 0; i < IM_ARRAYSIZE(tonemaps); i++) {
                    const bool is_selected = (tonemapper == i);
                    if (ImGui::Selectable(tonemaps[i], is_selected)) {
                        tonemapper = i;
                        updateTexture = true;
                    }

                    // Set the initial focus when opening the combo (scrolling + keyboard navigation focus).
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            helpMarker("Tonemapping function applied to displayed image");
            if (ImGui::SliderFloat("exposure", &exposure, -25.0f, 25.0f, "%.1f")) {
                updateTexture = true;
            }
            ImGui::SameLine();
            helpMarker("Multiplication factor of displayed image values");
            ImGui::SliderFloat("zoom", &zoom, 0.1f, 10.0f, "%.1fx");
            ImGui::SameLine();
            helpMarker("Multiplication factor of displayed image size");
            if (displayError) {
                ImGui::BeginDisabled();
            }
            if (ImGui::Checkbox("path traced result", &displayPathTrace)) {
                updateTexture = true;
            }
            ImGui::SameLine();
            helpMarker("Toggle between reference and path tracer");
            if (displayError) {
                ImGui::EndDisabled();
            }
            if (ImGui::Checkbox("show error", &displayError)) {
                updateTexture = true;
            }
            ImGui::SameLine();
            helpMarker("Error between reference and path tracer");
            // Display section end
            ImGui::Dummy(ImVec2(0.0f, 1.0f));
            ImGui::Separator();
            if (!rendered || rendering) {
                ImGui::EndDisabled();
            }

            /////////////////////////////////////////////
            // Save section
            /////////////////////////////////////////////

            // Save section begin
            if (!rendered || rendering) {
                ImGui::BeginDisabled(true);
            }
            ImGui::Text("Save:");

            // Output file selection
            if (ImGui::Button(outputName.c_str(), ImVec2(ImGui::CalcItemWidth(), 20))) {
                fileDialogSave.Open();
            }
            ImGui::SameLine();
            ImGui::Text("file");
            ImGui::SameLine();
            helpMarker("Output EXR file");
            fileDialogSave.Display();
            if (fileDialogSave.HasSelected()) {
                outputPath = fileDialogSave.GetSelected().string();
                outputName = outputPath.substr(outputPath.find_last_of(dirSeparator) + 1);
                fileDialogSave.ClearSelected();
            }

            // Save button
            if (ImGui::Button("Save")) {
                const char* err = NULL;
                const int   ret = SaveEXR(result[0].data(),
                                        renderedResolution,
                                        renderedResolution,
                                        3,
                                        0,
                                        outputPath.c_str(),
                                        &err);
                if (ret != TINYEXR_SUCCESS) {
                    saveError = std::string(err);
                    saved     = false;
                } else {
                    saved = true;
                }
                FreeEXRErrorMessage(err);
            }
            if (saved) {
                ImGui::SameLine();
                ImGui::Text("OK");
            } else if (!saveError.empty()) {
                ImGui::SameLine();
                errorMarker(saveError.c_str());
            }

            // Save section end
            if (!rendered || rendering) {
                ImGui::EndDisabled();
            }

            /////////////////////////////////////////////
            // Help
            /////////////////////////////////////////////

            ImVec2 bottomPos = ImVec2(5, ImGui::GetWindowSize().y - 35.f);
            ImGui::SetCursorPos(bottomPos);
            if (ImGui::Button("?", ImVec2(30.f, 30.f))) {
                ImGui::OpenPopup("helpPopup");
            }
            ImGui::SameLine();
            if (ImGui::BeginPopup("helpPopup")) {
                ImGui::Text("Instructions:");
                ImGui::Indent();
                ImGui::Text("1. Select model dataset file in the 'Dataset' section and hit 'Load'.");
                ImGui::Text("2. Select sky parameters you wish to render in the 'Configuration' section and "
                            "hit 'Render'. The result will show up on the right.");
                ImGui::Indent();
                ImGui::Text("You can also check 'Auto-update' and the rendered image will update "
                            "automatically with any change in the 'Configuration' section.");
                ImGui::Unindent();
                ImGui::Indent();
                ImGui::Text("Note: You can use CTRL + Click on the sliders to input values directly.");
                ImGui::Unindent();
                ImGui::Text("3. Select whether to display all visible channels combined in one sRGB image or "
                            "individual channels in the 'Channels' section.");
                ImGui::Text("4. Modify the way the result is displayed in the 'Display' section.");
                ImGui::Text("5. Save the result to a file in the 'Save' section.");
                ImGui::Indent();
                ImGui::Text(
                    "Note: The result being saved is not affected by settings from the 'Display' section.");
                ImGui::Unindent();
                ImGui::Unindent();
                ImGui::EndPopup();
            }

            ImGui::End();
        }

        // Output window
        {
            ImGui::SetNextWindowPos(ImVec2(windowWidthInput, 0));
            ImGui::SetNextWindowSize(ImVec2(windowWidthImage, windowHeightImage));
            ImGui::Begin("Output", NULL, outputWindowFlags);

            // Update the texture if needed
            if (updateTexture) {
                if (displayError) {

                } else {
                    if (displayPathTrace) {
                        convertToTexture(resultPt, renderedResolution, exposure, tonemapper, &texture);
                    } else {
                        convertToTexture(&result[0][0], renderedResolution, exposure, tonemapper, &texture);
                    }
                }
            }

            // Display the texture in the center of the window
            ImVec2 diplaySize = ImVec2(renderedResolution * zoom, renderedResolution * zoom);
            ImVec2 windowSize = ImGui::GetWindowSize();
            ImVec2 initialPos = ImGui::GetCursorPos();
            ImVec2 centerPos =
                ImVec2((windowSize.x - diplaySize.x) * 0.5f, (windowSize.y - diplaySize.y) * 0.5f);
            centerPos = ImVec2(std::max(centerPos.x, initialPos.x), std::max(centerPos.y, initialPos.y));
            ImGui::SetCursorPos(centerPos);
            if (texture) {
                ImGui::Image(texture, diplaySize);
            }

            ImGui::End();
        }

        updateTexture = false;

        // Frame rendering
        ImGui::Render();
#ifdef _WIN32
        const float clear_color_with_alpha[4] = { 0.f, 0.f, 0.f, 1.f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        g_pSwapChain->Present(1, 0);
#else
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
#endif
    }

    // Cleanup
#ifdef _WIN32
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    cleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);
#else
    ImGui_ImplOpenGL2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
#endif

    return 0;
}