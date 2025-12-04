#include "MyApp.h"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <numeric>
#include <random>

#include <imgui.h>

enum class AssetType { Asset, Kernel, Shader };

namespace {
	const std::filesystem::path rootPath = "../../../src/opencl_06_opengl_nbody";

	template <AssetType T>
	std::string PathTo(const std::string& filename) {
		std::filesystem::path result;

		if constexpr (T == AssetType::Asset)
			result = rootPath / "assets" / filename;
		else if constexpr (T == AssetType::Kernel)
			result = rootPath / "kernels" / filename;
		else if constexpr (T == AssetType::Shader)
			result = rootPath / "shaders" / filename;
		else
			result = rootPath / filename;

		if (!std::filesystem::exists(result))
			throw std::runtime_error("File not found: " + result.string());

		return result.string();
	}

	// Keep some history for smoother graphs
	constexpr std::size_t MAX_HISTORY = 120;
	std::deque<float> frameTimes;
	std::deque<float> kernelTimes;

	void addSample(std::deque<float>& buffer, float value) {
		if (buffer.size() >= MAX_HISTORY)
			buffer.pop_front();
		buffer.push_back(value);
	}

	float average(const std::deque<float>& buffer) {
		if (buffer.empty()) return 0.0f;
		return std::accumulate(buffer.begin(), buffer.end(), 0.0f) / buffer.size();
	}
}

MyApp::MyApp() = default;
MyApp::~MyApp() = default;

void MyApp::InitGL() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// Depth test
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// Create vertex buffer for particles
	vbo = createBuffer();
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, maxParticles * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Create vertex buffer for velocities
	vboVel = createBuffer();
	glBindBuffer(GL_ARRAY_BUFFER, *vboVel);
	glBufferData(GL_ARRAY_BUFFER, maxParticles * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Create vertex array object to handle vertex properties during rendering
	vao = createVertexArray();
	glBindVertexArray(*vao);

	// Positions
	glBindBuffer(GL_ARRAY_BUFFER, *vbo); // Attach VBO to VAO
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	// Velocities
	glBindBuffer(GL_ARRAY_BUFFER, *vboVel); // Attach VBO to VAO
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);

	// Setup particle shader
	shaderProgram.AttachShader(GL_VERTEX_SHADER, PathTo<AssetType::Shader>("particle.vert"));
	shaderProgram.AttachShader(GL_GEOMETRY_SHADER, PathTo<AssetType::Shader>("particle.geom"));
	shaderProgram.AttachShader(GL_FRAGMENT_SHADER, PathTo<AssetType::Shader>("particle.frag"));
	shaderProgram.BindAttribLoc(0, "vs_in_pos");
	shaderProgram.BindAttribLoc(1, "vs_in_vel");
	if (!shaderProgram.LinkProgram())
		throw std::runtime_error("Failed to Link shader program.");

	// Load particle texture
	const auto image = ImageFromFile(PathTo<AssetType::Asset>("particle.png"));
	particleTexture = createTexture<GL_TEXTURE_2D>();
	glTextureStorage2D(*particleTexture, NumberOfMIPLevels(image), GL_RGBA8, image.width, image.height);
	glTextureSubImage2D(*particleTexture, 0, 0, 0, image.width, image.height, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
	glGenerateTextureMipmap(*particleTexture);
	glTextureParameteri(*particleTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTextureParameteri(*particleTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	float maxAnisotropy = 1.0f;
	glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &maxAnisotropy);
	glTextureParameterf(*particleTexture, GL_TEXTURE_MAX_ANISOTROPY, maxAnisotropy);

	// Init Camera
	float aspect = (windowHeight > 0)
		? static_cast<float>(windowWidth) / static_cast<float>(windowHeight)
		: 1.0f;

	proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);
	view = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 1.0f),   // kamera pozíció
		glm::vec3(0.0f, 0.0f, 0.0f),   // hova néz
		glm::vec3(0.0f, 1.0f, 0.0f)    // felfelé
	);
}

void MyApp::InitCL() {
	if (!oclCreateContextFromCurrentGLContext(context))
		throw cl::Error(CL_INVALID_CONTEXT, "Failed to create shared CL/GL context");

	const auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto device = devices.front();
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
	queue = cl::CommandQueue(context, device);

	// Build OpenCL program
	const auto sourceCode = oclReadSourcesFromFile(PathTo<AssetType::Kernel>("GLinterop.cl"));
	program = cl::Program(context, sourceCode);
	try {
		program.build(devices);
	}
	catch (const cl::Error&) {
		for (auto&& [dev, log] : program.getBuildInfo<CL_PROGRAM_BUILD_LOG>())
			std::cerr << "Build log for " << dev.getInfo<CL_DEVICE_NAME>() << ":\n" << log << "\n";
		throw;
	}
	// Init kernels
	kernelCellIndex = cl::Kernel(program, "computeParticleCellIndex");
	kernelComputeCOM = cl::Kernel(program, "computeCellCOM");
	kernelUpdate = cl::Kernel(program, "update");

	// Shared GL/CL buffer
	clVboBuffer = cl::BufferGL(context, CL_MEM_WRITE_ONLY, *vbo);
	clMasses = cl::Buffer(context, CL_MEM_READ_WRITE, maxParticles * sizeof(float));
	clVelocities = cl::BufferGL(context, CL_MEM_READ_WRITE, *vboVel);

	// Init Grid + COM buffers
	clParticleCellIndex = cl::Buffer(context, CL_MEM_READ_WRITE, maxParticles * sizeof(int));
	clCellCOM = cl::Buffer(context, CL_MEM_READ_WRITE, gridNx * gridNy * sizeof(glm::vec3));
	clCellMass = cl::Buffer(context, CL_MEM_READ_WRITE, gridNx * gridNy * sizeof(float));

	// Set kernel arguments

	kernelCellIndex.setArg(0, clVboBuffer);
	kernelCellIndex.setArg(1, clParticleCellIndex);
	kernelCellIndex.setArg(2, gridNx);
	kernelCellIndex.setArg(3, gridNy);
	kernelCellIndex.setArg(4, gridNz);
	kernelCellIndex.setArg(5, cellSizeInvX);
	kernelCellIndex.setArg(6, cellSizeInvY);
	kernelCellIndex.setArg(7, cellSizeInvZ);
	kernelCellIndex.setArg(8, worldMinX);
	kernelCellIndex.setArg(9, worldMinY);
	kernelCellIndex.setArg(10, worldMinZ);
	kernelCellIndex.setArg(11, currentNumParticles);

	kernelComputeCOM.setArg(0, clVboBuffer);
	kernelComputeCOM.setArg(1, clMasses);
	kernelComputeCOM.setArg(2, clParticleCellIndex);
	kernelComputeCOM.setArg(3, clCellMass);
	kernelComputeCOM.setArg(4, clCellCOM);
	kernelComputeCOM.setArg(5, currentNumParticles);
	kernelComputeCOM.setArg(6, totalCells);
	kernelComputeCOM.setArg(7, cl::Local(localSize * sizeof(float)));
	kernelComputeCOM.setArg(8, cl::Local(localSize * sizeof(float)));
	kernelComputeCOM.setArg(9, cl::Local(localSize * sizeof(float)));
	kernelComputeCOM.setArg(10, cl::Local(localSize * sizeof(float)));

	kernelUpdate.setArg(0, clVboBuffer);
	kernelUpdate.setArg(1, clVelocities);
	kernelUpdate.setArg(2, clMasses);
	kernelUpdate.setArg(3, clParticleCellIndex);
	kernelUpdate.setArg(4, clCellMass);
	kernelUpdate.setArg(5, clCellCOM);
	kernelUpdate.setArg(6, gridNx);
	kernelUpdate.setArg(7, gridNy);
	kernelUpdate.setArg(8, gridNz);
	kernelUpdate.setArg(9, totalCells);
	kernelUpdate.setArg(10, currentNumParticles);

	ResetSimulation();
}

void MyApp::ResetSimulation() {
	currentNumParticles = numParticles;
	// Initialize particle data
	std::vector<float> masses(currentNumParticles, 1.f);
	queue.enqueueWriteBuffer(clMasses, CL_TRUE, 0, masses.size() * sizeof(float), masses.data());

	std::vector<glm::vec3> velocities(currentNumParticles, glm::vec3{});
	if (useRandomVelocities)
		for (size_t i = 0; i < velocities.size(); i += 2) {
			double angle = i / double(velocities.size() / 2) * (2 * M_PI);
			velocities[i].x = static_cast<float>(-std::cos(angle) * 1.7);
			velocities[i].y = static_cast<float>(std::sin(angle) * 1.7);
			velocities[i].z = static_cast<float>(std::sin(angle) * 0.3);
		}

	// Initialize positions
	std::vector<glm::vec3> positions(currentNumParticles);
	std::mt19937 rng(std::random_device{}());
	switch (initDistribution) {
	default:
	case 0: {
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
		std::generate(positions.begin(), positions.end(), [&] { return glm::vec3{ dist(rng), dist(rng), dist(rng) }; });
		break;
	}
	case 1: {
		for (int i = 0; i < currentNumParticles; ++i) {
			float angle = (static_cast<float>(i) / currentNumParticles) * 2.0f * M_PI;
			float r = 0.25f;
			positions[i] = glm::vec3(r * std::sin(angle), r * std::cos(angle), r * std::sin(3.0f * angle));
		}
		break;
	}
	case 2: {
		glm::vec3 A(-0.6f, -0.5f, -0.2f);
		glm::vec3 B(0.6f, -0.5f, -0.2f);
		glm::vec3 C(0.0f, 0.6f, 0.4f);

		std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

		for (int i = 0; i < currentNumParticles; ++i) {
			float u = dist01(rng);
			float v = dist01(rng);
			if (u + v > 1.0f) {
				u = 1.0f - u;
				v = 1.0f - v;
			}
			glm::vec3 P = A + u * (B - A) + v * (C - A);
			positions[i] = glm::vec3(P);
		}
		break;
	}
	case 3: {
		std::normal_distribution<float> gauss(0.0f, 0.25f);
		for (int i = 0; i < currentNumParticles; ++i) {
			float x = gauss(rng);
			float y = gauss(rng);
			float z = gauss(rng);
			positions[i] = glm::vec3(x, y, z);
		}
		break;
		break;
	}
	case 4: {
		std::normal_distribution<float> noise(0.0f, 0.02f);
		const float arms = static_cast<float>(spiralArms);
		for (int i = 0; i < currentNumParticles; ++i) {
			float t = static_cast<float>(i) / currentNumParticles;
			float angle = t * arms * 6.0f * static_cast<float>(M_PI);
			float radius = 0.05f + 0.45f * t;

			float x = std::cos(angle) * radius + noise(rng);
			float y = std::sin(angle) * radius + noise(rng);
			float z = 0.15f * std::sin(angle * 0.5f) * (1.0f - t);

			positions[i] = glm::vec3(x, y, z);
		}
		break;
	}
	}

	// Positions upload
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	if (glm::vec3* ptr = static_cast<glm::vec3*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY))) {
		std::copy(positions.begin(), positions.end(), ptr);
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Velocities upload
	glBindBuffer(GL_ARRAY_BUFFER, *vboVel);
	if (glm::vec3* vptr = static_cast<glm::vec3*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY))) {
		std::copy(velocities.begin(), velocities.end(), vptr);
		glUnmapBuffer(GL_ARRAY_BUFFER);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	kernelCellIndex.setArg(11, currentNumParticles);
	kernelComputeCOM.setArg(5, currentNumParticles);
	kernelUpdate.setArg(10, currentNumParticles);
}

void MyApp::Update(const UpdateInfo& info) {
	if (!simulation_paused) {
		float deltaTime = std::clamp(info.deltaTimeSec, 0.0000001f, 0.001f);
		kernelUpdate.setArg(11, gravityConstant);
		kernelUpdate.setArg(12, deltaTime);

		std::vector<cl::Memory> glObjects{ clVboBuffer, clVelocities };
		queue.enqueueAcquireGLObjects(&glObjects);
		queue.enqueueNDRangeKernel(kernelCellIndex, cl::NullRange, cl::NDRange(globalParticles), cl::NDRange(localSize));

		queue.enqueueNDRangeKernel(kernelComputeCOM, cl::NullRange, cl::NDRange(globalCOM), cl::NDRange(localSize));

		queue.enqueueNDRangeKernel(kernelUpdate, cl::NullRange, cl::NDRange(globalParticles), cl::NDRange(localSize));
		queue.enqueueReleaseGLObjects(&glObjects);
		queue.finish();
	}

	addSample(frameTimes, info.deltaTimeSec * 1000);
	addSample(kernelTimes, SDL_GetTicks() - info.elapsedTimeSec * 1000);
}

void MyApp::Render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	shaderProgram.On();
	shaderProgram.SetUniform("particle_size", particleSize);
	shaderProgram.SetTexture("tex0", 0, *particleTexture);

	glm::mat4 viewProj = proj * view;
	shaderProgram.SetUniform("u_viewProj", viewProj);

	glBindVertexArray(*vao);
	glDrawArrays(GL_POINTS, 0, currentNumParticles);
	glBindVertexArray(0);

	shaderProgram.Off();
}

void MyApp::RenderGUI()
{
	// ImGui rendering logic would go here
	//ImGui::ShowDemoWindow();

	ImGui::Begin("Performance");

	// === FPS ===
	float avgFrame = average(frameTimes);
	float fps = avgFrame > 0.0f ? 1000.0f / avgFrame : 0.0f;
	ImGui::Text("Frame time: %.2f ms (%.1f FPS)", avgFrame, fps);
	auto vec_frameTimes = std::vector<float>(frameTimes.begin(), frameTimes.end());
	ImGui::PlotLines("Frame Time (ms)", vec_frameTimes.data(),
		static_cast<int>(frameTimes.size()), 0, nullptr, 0.0f, 50.0f,
		ImVec2(0, 60));

	// === Kernel timing ===
	ImGui::Separator();
	float avgKernel = average(kernelTimes);
	ImGui::Text("Kernel time: %.3f ms (avg over %zu frames)", avgKernel, kernelTimes.size());
	auto vec_kernelTimes = std::vector<float>(kernelTimes.begin(), kernelTimes.end());
	ImGui::PlotLines("Kernel Time (ms)", vec_kernelTimes.data(),
		static_cast<int>(kernelTimes.size()), 0, nullptr, 0.0f, avgKernel * 3.0f,
		ImVec2(0, 60));

	// === Compute–Render Ratio ===
	ImGui::Separator();
	if (avgFrame > 0.0f)
		ImGui::Text("GPU compute load: %.1f%%", (avgKernel / avgFrame) * 100.0f);

	// === Controls ===
	ImGui::Separator();
	ImGui::Text("Parameters");
	ImGui::SliderFloat(
		"G (gravity)",
		&gravityConstant,
		1e-6f,
		5e-3f,
		"%.6f"
	);
	ImGui::SliderInt(
		"Number of particles",
		&numParticles,
		2,
		maxParticles
	);
	ImGui::Separator();
	ImGui::Text("Initial distribution");
	ImGui::RadioButton("Uniform random", &initDistribution, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Ring", &initDistribution, 1);
	ImGui::SameLine();
	ImGui::RadioButton("Triangle", &initDistribution, 2);
	ImGui::SameLine();
	ImGui::RadioButton("Gaussian blob", &initDistribution, 3);
	ImGui::SameLine();
	ImGui::RadioButton("Spiral galaxy", &initDistribution, 4);
	if (initDistribution == 4) {
		ImGui::SliderInt("Spiral arms", &spiralArms, 1, 2);
	}

	ImGui::Separator();
	ImGui::Text("Simulation Controls");
	ImGui::Checkbox("Pause Simulation", &simulation_paused);
	if (ImGui::Button("Reset simulation")) {
		ResetSimulation();
	}

	ImGui::End();
}

void MyApp::KeyboardDown(const SDL_KeyboardEvent&) {}
void MyApp::KeyboardUp(const SDL_KeyboardEvent&) {}
void MyApp::MouseMove(const SDL_MouseMotionEvent&) {}
void MyApp::MouseDown(const SDL_MouseButtonEvent&) {}
void MyApp::MouseUp(const SDL_MouseButtonEvent&) {}
void MyApp::MouseWheel(const SDL_MouseWheelEvent&) {}
void MyApp::OtherEvent(const SDL_Event&) {}

void MyApp::Resize(int width, int height) {
	glViewport(0, 0, width, height);
	windowWidth = width;
	windowHeight = height;

	float aspect = (height > 0)
		? static_cast<float>(width) / static_cast<float>(height)
		: 1.0f;

	proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);
}