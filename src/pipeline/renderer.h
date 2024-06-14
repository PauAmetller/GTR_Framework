#pragma once
#include "scene.h"
#include "prefab.h"

#include "light.h"
#include "../gfx/sphericalharmonics.h"

#define NUM_SHADOW_MAPS 10

//forward declarations
class Camera;
class Skeleton;
namespace GFX {
	class Shader;
	class Mesh;
	class FBO;
}

enum ePipelineMode {
	FLAT,
	FORWARD,
	DEFERRED,
	PIPELINE_COUNT
};

enum eShowGBuffer {
	NONE,
	COLOR,
	NORMAL,
	EMISSIVE,
	DEPTH,
	EXTRA,
	GBUFFERS,
	GBUFFERS_COUNT
};

enum eSSAOMODE {
	SSAO,
	SSAO_PLUS,
	SSAO_COUNT
};

//struct to store probes
struct sProbe {
	vec3 pos; //where is located
	vec3 local; //its ijk pos in the matrix
	int index; //its index in the linear array
	SphericalHarmonics sh; //coeffs
};

//struct to store grid info
struct sIrradianceInfo {
	vec3 start;
	vec3 end;
	vec3 dim;
	vec3 delta;
	int num_probes;
};

struct sReflectionProbe {
	vec3 pos;
	GFX::Texture* texture = nullptr;
};

namespace SCN {

	class Prefab;
	class Material;

	class Renderable {  ////////////////
	public:
		GFX::Mesh* mesh;
		SCN::Material* material;
		Matrix44 model;
		BoundingBox bounding;
		float distance_to_camera;

	};///////////


	// This class is in charge of rendering anything in our system.
	// Separating the render from anything else makes the code cleaner
	class Renderer
	{
	public:
		bool render_wireframe;
		bool render_boundaries;
		bool deactivate_ambient_light;
		bool albedo_texture;
		bool emissive_texture;
		bool occlusion_texture;
		bool metallicRoughness_texture;
		bool normalMap_texture;
		bool white_textures;
		bool skip_lights;
		bool skip_shadows;
		bool skip_alpha_renderables;
		bool Remove_PBR;
		bool show_ssao;
		bool ssao_texture;
		bool blurr;
		bool deactivate_tonemapper;
		bool Linear_space;
		bool probes_grid;
		bool reflection_probes_grid;
		bool volumetric_light;
		bool decal;

		float ssao_max_distance;

		int shadow_map_index;
		int shadow_map_size;
		int power_of_two_shadowmap = 10; //exponent for shadow_map_size
		int irradiance_capture_size;
		int power_of_two_irradiance = 4; //exponent for irradiance_capture_size

		//Volumetric
		float air_density;
		float weight_ambient_light;

		//SSAO blur
		int kernel_size;
		float sigma;

		//Tonemapper
		float scale; //color scale before tonemapper
		float average_lum;
		float lumwhite2;
		float igamma;

		ePipelineMode pipeline_mode;
		eShowGBuffer show_gbuffer;
		eProcessing processing_mode;
		eSSAOMODE ssao_mode;

		GFX::Texture* skybox_cubemap;
		GFX::FBO* shadow_maps[NUM_SHADOW_MAPS];

		SCN::Scene* scene;

		std::vector<Renderable> renderables; ///////////
		std::vector<Renderable> opaqueRenderables;
		std::vector<Renderable> alphaRenderables;
		std::vector<LightEntity*> lights; //////////
		std::vector<LightEntity*> directional_lights; //////////
		std::vector<LightEntity*> point_and_spot_lights; //////////
		LightEntity* moon_light; ////////
		std::vector<DecalEntity*> decals;

		//SSAO
		float ssao_radius;
		float ssao_linear;

		//updated every frame
		Renderer(const char* shaders_atlas_filename );

		//just to be sure we have everything ready for the rendering
		void setupScene();

		//add here your functions////////////////
		void extractRenderables(SCN::Node* node, Camera* camera);
		void extractSceneInfo(SCN::Scene* scene, Camera* camera);
		void generateShadowMaps(Camera* main_camera);
		/////////////

		//renders several elements of the scene
		void renderScene(SCN::Scene* scene, Camera* camera);
		void renderSceneForward(SCN::Scene* scene, Camera* camera);
		void renderSceneDeferred(SCN::Scene* scene, Camera* camera);

		//render the skybox
		void renderSkybox(GFX::Texture* cubemap);
	
		//to render one node from the prefab and its children
		void renderNode(SCN::Node* node, Camera* camera);

		void renderMeshWithMaterialFlat(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		void renderMeshWithMaterialGBuffers(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		//to render one mesh given its material and transformation matrix
		void renderMeshWithMaterial(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material);

		//to render one mesh given its material and transformation matrix, with lights
		void renderMeshWithMaterialLights(const Matrix44 model, GFX::Mesh* mesh, SCN::Material* material); ///////////

		void renderProbe(vec3 pos, float scale, SphericalHarmonics& shs);
		void renderProbes(float scale);
		void captureProbe(sProbe& p);
		void captureProbes();

		void renderReflectionProbe(sReflectionProbe* p, float scale);
		void renderReflectionProbes(float scale);
		void captureReflectionProbe(sReflectionProbe* p);
		void captureReflectionProbes();
		void capturePlanarReflection(Camera* camera);

		void showUI();

		void cameraToShader(Camera* camera, GFX::Shader* shader); //sends camera uniforms to shader
		void lightToShader(LightEntity* light, GFX::Shader* shader); //sends camera uniforms to shader
		void GbuffersToShader(GFX::FBO* gbuffers, GFX::Shader* shader); //sends the gbuffers uniforms to shader
	};

};