#pragma once
#define _USE_MATH_DEFINES

// GLM
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL3/SDL.h>
#include <SDL3/SDL_opengl.h>

// Utils
#include "gShaderProgram.h"
#include <GLUtils.hpp>

// OpenCL
#include <CL/opencl.hpp>
#include <oclutils.hpp>
#include <oglutils.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct UpdateInfo {
	float elapsedTimeSec = 0.0f; // Total time since program start
	float deltaTimeSec = 0.0f; // Time since last update
};

class MyApp {
public:
	MyApp();
	~MyApp();

	void InitGL();
	void InitCL();

	void Update(const UpdateInfo& info);
	void Render();
	void RenderGUI();

	// SDL Event Handlers
	void KeyboardDown(const SDL_KeyboardEvent&);
	void KeyboardUp(const SDL_KeyboardEvent&);
	void MouseMove(const SDL_MouseMotionEvent&);
	void MouseDown(const SDL_MouseButtonEvent&);
	void MouseUp(const SDL_MouseButtonEvent&);
	void MouseWheel(const SDL_MouseWheelEvent&);
	void Resize(int width, int height);
	void OtherEvent(const SDL_Event&);

	void ResetSimulation();

private:
	// Window
	int windowWidth = 0;
	int windowHeight = 0;

	// Camera
	glm::mat4 view = glm::mat4(1.0f);
	glm::mat4 proj = glm::mat4(1.0f);

	// Grids (3D)
	int gridNx = 32;
	int gridNy = 32;
	int gridNz = 32;
	int totalCells = gridNx * gridNy * gridNz;

	// World size
	float worldMinX = -1.0f;
	float worldMaxX = 1.0f;
	float worldMinY = -1.0f;
	float worldMaxY = 1.0f;
	float worldMinZ = -1.0f;
	float worldMaxZ = 1.0f;

	// Grid sizes
	float cellSizeX = (worldMaxX - worldMinX) / gridNx;
	float cellSizeY = (worldMaxY - worldMinY) / gridNy;
	float cellSizeZ = (worldMaxZ - worldMinZ) / gridNz;
	float cellSizeInvX = 1.0f / cellSizeX;
	float cellSizeInvY = 1.0f / cellSizeY;
	float cellSizeInvZ = 1.0f / cellSizeZ;
	float cellSize = fmax(fmax(cellSizeX, cellSizeY), cellSizeZ);

	// OpenGL
	UniqueGlVertexArray vao;
	UniqueGlBuffer      vbo;
	UniqueGlBuffer      vboVel; // Buffer for velocities
	UniqueGlTexture     particleTexture;
	gShaderProgram      shaderProgram;

	// OpenCL
	cl::Context       context;
	cl::CommandQueue  queue;
	cl::Program       program;
	cl::Kernel        kernelUpdate;

	// New kernels for grid and COM
	cl::Kernel        kernelCellIndex;
	cl::Kernel		  kernelComputeCOM;

	cl::BufferGL      clVboBuffer;
	cl::BufferGL      clVelocities;
	cl::Buffer        clMasses;

	// Grid buffer
	cl::Buffer clParticleCellIndex;

	// COM buffers
	cl::Buffer clCellMass;
	cl::Buffer clCellCOM;

	// Simulation parameters
	static constexpr float particleSize = 0.01f;
	static constexpr bool  useRandomVelocities = true;
	static constexpr float massiveObjectMass = 1.0f;

	// ImGui
	static constexpr int maxParticles = 50000;  // buffer capacity
	int numParticles = 20000;
	int currentNumParticles = 20000;
	float gravityConstant = 0.0001f;

	// Initial distribution type (0..4)
	// 0 = Uniform random
	// 1 = Ring
	// 2 = Triangle
	// 3 = Gaussian blob
	// 4 = Spiral galaxy
	int initDistribution = 0;

	// extra parameter for Spiral galaxy initial distribution (1..4)
	int spiralArms = 2;

	// GPU Optimization helpers
	const size_t localSize = 128;
	const size_t globalParticles = ((size_t)maxParticles + localSize - 1) / localSize * localSize;
	const size_t globalCOM = ((size_t)totalCells) * localSize;

	// Application state
	bool simulation_paused = false;
};
