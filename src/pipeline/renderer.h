#pragma once
#include "scene.h"
#include "prefab.h"

#include "light.h"

#define NUM_SHADOW_MAPS 4

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
	DEPTH,
	EXTRA,
	GBUFFERS_COUNT
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

		int shadow_map_index;
		int shadow_map_size;
		int power_of_two = 10; //exponent for shadow_map_size

		ePipelineMode pipeline_mode;
		eShowGBuffer show_gbuffer;

		GFX::Texture* skybox_cubemap;
		GFX::FBO* shadow_maps[NUM_SHADOW_MAPS];

		SCN::Scene* scene;

		std::vector<Renderable> renderables; ///////////
		std::vector<Renderable> opaqueRenderables;
		std::vector<Renderable> alphaRenderables;
		std::vector<LightEntity*> lights; //////////
		LightEntity* moon_light; ////////

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

		void showUI();

		void cameraToShader(Camera* camera, GFX::Shader* shader); //sends camera uniforms to shader
		void lightToShader(LightEntity* light, GFX::Shader* shader); //sends camera uniforms to shader
	};

};